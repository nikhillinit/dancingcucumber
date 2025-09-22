import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { openaiService } from "./services/openai";
import { financialDataService } from "./services/financial-data";
import { newsService } from "./services/news";
import { 
  insertStockSchema, 
  insertStockAnalysisSchema, 
  insertDebateSchema, 
  insertPortfolioPositionSchema,
  insertChatConversationSchema,
  insertChatMessageSchema,
  insertPortfolioUploadSchema
} from "@shared/schema";
import { ObjectStorageService, ObjectNotFoundError } from "./objectStorage";

export async function registerRoutes(app: Express): Promise<Server> {
  // Personas endpoints
  app.get("/api/personas", async (req, res) => {
    try {
      const personas = await storage.getPersonas();
      res.json(personas);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch personas" });
    }
  });

  // Stocks endpoints
  app.get("/api/stocks", async (req, res) => {
    try {
      const stocks = await storage.getStocks();
      res.json(stocks);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch stocks" });
    }
  });

  app.get("/api/stocks/search", async (req, res) => {
    try {
      const { q } = req.query;
      if (!q || typeof q !== 'string') {
        return res.status(400).json({ message: "Search query required" });
      }

      const results = await financialDataService.searchStocks(q);
      res.json(results);
    } catch (error) {
      res.status(500).json({ message: "Failed to search stocks" });
    }
  });

  app.get("/api/stocks/:symbol", async (req, res) => {
    try {
      const { symbol } = req.params;
      
      // Try to get from storage first
      let stock = await storage.getStockBySymbol(symbol.toUpperCase());
      
      if (!stock) {
        // Fetch from external API and create
        const stockData = await financialDataService.getStockQuote(symbol);
        const financialMetrics = await financialDataService.getFinancialMetrics(symbol).catch(() => null);
        
        stock = await storage.createStock({
          symbol: stockData.symbol,
          name: stockData.name,
          currentPrice: stockData.price.toString(),
          priceChange: stockData.change.toString(),
          priceChangePercent: stockData.changePercent.toString(),
          marketData: {
            ...stockData,
            ...(financialMetrics ? { financialMetrics } : {})
          }
        });
      } else {
        // Update with fresh data
        const stockData = await financialDataService.getStockQuote(symbol);
        stock = await storage.updateStock(stock.id, {
          currentPrice: stockData.price.toString(),
          priceChange: stockData.change.toString(),
          priceChangePercent: stockData.changePercent.toString(),
          marketData: stockData
        });
      }

      res.json(stock);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch stock data" });
    }
  });

  // Stock analysis endpoints
  app.post("/api/stocks/:symbol/analyze", async (req, res) => {
    try {
      const { symbol } = req.params;
      
      // Get or create stock
      let stock = await storage.getStockBySymbol(symbol.toUpperCase());
      if (!stock) {
        const stockData = await financialDataService.getStockQuote(symbol);
        stock = await storage.createStock({
          symbol: stockData.symbol,
          name: stockData.name,
          currentPrice: stockData.price.toString(),
          priceChange: stockData.change.toString(),
          priceChangePercent: stockData.changePercent.toString(),
          marketData: { ...stockData }
        });
      }

      // Get personas and analyze
      const personas = await storage.getPersonas();
      const newsArticles = await newsService.getStockNews(symbol);
      const newsContext = newsArticles.map(article => article.title);

      const analyses = [];
      for (const persona of personas) {
        try {
          const analysis = await openaiService.analyzeStockByPersona(
            persona,
            stock,
            stock.marketData,
            newsContext
          );

          const savedAnalysis = await storage.createAnalysis({
            stockId: stock.id,
            personaId: persona.id,
            recommendation: analysis.recommendation,
            confidenceScore: analysis.confidenceScore,
            reasoning: analysis.reasoning,
            targetPrice: analysis.targetPrice?.toString()
          });

          analyses.push({ persona, analysis: savedAnalysis });
        } catch (error) {
          console.error(`Failed to analyze with ${persona.name}:`, error);
        }
      }

      // Generate consensus
      const consensusResult = await openaiService.generateConsensus(
        stock,
        analyses.map(({ persona, analysis }) => ({
          persona,
          analysis: {
            recommendation: analysis.recommendation as any,
            confidenceScore: analysis.confidenceScore,
            reasoning: analysis.reasoning,
            targetPrice: analysis.targetPrice ? parseFloat(analysis.targetPrice) : undefined
          }
        }))
      );

      res.json({
        stock,
        analyses,
        consensus: consensusResult
      });
    } catch (error) {
      console.error("Analysis error:", error);
      res.status(500).json({ message: "Failed to analyze stock" });
    }
  });

  app.get("/api/stocks/:symbol/analyses", async (req, res) => {
    try {
      const { symbol } = req.params;
      const stock = await storage.getStockBySymbol(symbol.toUpperCase());
      
      if (!stock) {
        return res.status(404).json({ message: "Stock not found" });
      }

      const analyses = await storage.getAnalysesByStock(stock.id);
      const enrichedAnalyses = [];

      for (const analysis of analyses) {
        const persona = await storage.getPersona(analysis.personaId);
        if (persona) {
          enrichedAnalyses.push({
            ...analysis,
            persona
          });
        }
      }

      res.json(enrichedAnalyses);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch analyses" });
    }
  });

  app.get("/api/analyses/latest", async (req, res) => {
    try {
      const limit = parseInt(req.query.limit as string) || 10;
      const analyses = await storage.getLatestAnalyses(limit);
      res.json(analyses);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch latest analyses" });
    }
  });

  // Portfolio endpoints
  app.get("/api/portfolio", async (req, res) => {
    try {
      const positions = await storage.getPortfolioPositions();
      
      // Calculate portfolio summary
      let totalValue = 0;
      let totalReturn = 0;
      let totalReturnPercent = 0;
      let totalConsensusScore = 0;

      for (const position of positions) {
        const currentValue = parseFloat(position.currentValue || "0");
        const returnAmount = parseFloat(position.totalReturn || "0");
        const returnPercent = parseFloat(position.returnPercent || "0");
        const consensusScore = position.consensusScore || 0;

        totalValue += currentValue;
        totalReturn += returnAmount;
        totalReturnPercent += returnPercent;
        totalConsensusScore += consensusScore;
      }

      if (positions.length > 0) {
        totalReturnPercent = totalReturnPercent / positions.length;
        totalConsensusScore = totalConsensusScore / positions.length;
      }

      res.json({
        positions,
        summary: {
          totalValue,
          totalReturn,
          totalReturnPercent,
          consensusScore: totalConsensusScore
        }
      });
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch portfolio" });
    }
  });

  app.post("/api/portfolio", async (req, res) => {
    try {
      console.log("Portfolio POST request body:", req.body);
      const { symbol, shares, price } = req.body;
      
      console.log("Extracted values:", { symbol, shares, price });
      
      if (!symbol || !shares || !price) {
        console.log("Validation failed - missing fields");
        return res.status(400).json({ message: "Symbol, shares, and price are required" });
      }

      // Get or create stock
      let stock = await storage.getStockBySymbol(symbol.toUpperCase());
      if (!stock) {
        const stockData = await financialDataService.getStockQuote(symbol);
        stock = await storage.createStock({
          symbol: stockData.symbol,
          name: stockData.name,
          currentPrice: stockData.price.toString(),
          priceChange: stockData.change.toString(),
          priceChangePercent: stockData.changePercent.toString(),
          marketData: stockData
        });
      }

      // Calculate current value and return
      const currentPrice = parseFloat(stock.currentPrice || "0");
      const sharesNum = parseFloat(shares);
      const avgPrice = parseFloat(price);
      const currentValue = currentPrice * sharesNum;
      const totalReturn = currentValue - (avgPrice * sharesNum);
      const returnPercent = avgPrice > 0 ? ((currentPrice - avgPrice) / avgPrice) * 100 : 0;

      const position = await storage.createPortfolioPosition({
        stockId: stock.id,
        shares: shares.toString(),
        averagePrice: price.toString(),
        currentValue: currentValue.toString(),
        totalReturn: totalReturn.toString(),
        returnPercent: returnPercent.toString(),
        consensusScore: 75 // Default consensus score
      });

      res.json(position);
    } catch (error) {
      console.error("Portfolio error:", error);
      res.status(500).json({ message: "Failed to add to portfolio" });
    }
  });

  // Debates endpoints
  app.get("/api/debates", async (req, res) => {
    try {
      const debates = await storage.getDebates();
      res.json(debates);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch debates" });
    }
  });

  app.get("/api/stocks/:symbol/debates", async (req, res) => {
    try {
      const { symbol } = req.params;
      const stock = await storage.getStockBySymbol(symbol.toUpperCase());
      
      if (!stock) {
        return res.status(404).json({ message: "Stock not found" });
      }

      const debates = await storage.getDebatesByStock(stock.id);
      res.json(debates);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch debates" });
    }
  });

  app.post("/api/debates", async (req, res) => {
    try {
      const validation = insertDebateSchema.safeParse(req.body);
      if (!validation.success) {
        return res.status(400).json({ message: "Invalid debate data" });
      }

      const debate = await storage.createDebate(validation.data);
      res.json(debate);
    } catch (error) {
      res.status(500).json({ message: "Failed to create debate" });
    }
  });

  // News endpoints
  app.get("/api/news", async (req, res) => {
    try {
      const limit = parseInt(req.query.limit as string) || 20;
      const articles = await newsService.getMarketNews(limit);
      
      // Store in database for persistence
      for (const article of articles) {
        try {
          await storage.createNewsArticle({
            title: article.title,
            description: article.description,
            url: article.url,
            imageUrl: article.urlToImage,
            source: article.source.name,
            publishedAt: new Date(article.publishedAt),
            stockSymbols: [], // Would need NLP to extract stock symbols
            sentiment: "NEUTRAL",
            impact: "MEDIUM"
          });
        } catch (error) {
          // Article might already exist, ignore
        }
      }

      res.json(articles);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch news" });
    }
  });

  app.get("/api/news/stocks/:symbol", async (req, res) => {
    try {
      const { symbol } = req.params;
      const articles = await newsService.getStockNews(symbol);
      res.json(articles);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch stock news" });
    }
  });

  // Chat endpoints
  app.get("/api/chat/conversations", async (req, res) => {
    try {
      const conversations = await storage.getChatConversations();
      res.json(conversations);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch conversations" });
    }
  });

  app.post("/api/chat/conversations", async (req, res) => {
    try {
      const validation = insertChatConversationSchema.safeParse(req.body);
      if (!validation.success) {
        return res.status(400).json({ message: "Invalid conversation data" });
      }

      const conversation = await storage.createChatConversation(validation.data);
      res.json(conversation);
    } catch (error) {
      res.status(500).json({ message: "Failed to create conversation" });
    }
  });

  app.get("/api/chat/conversations/:id/messages", async (req, res) => {
    try {
      const { id } = req.params;
      const messages = await storage.getChatMessages(id);
      res.json(messages);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch messages" });
    }
  });

  app.post("/api/chat/conversations/:id/messages", async (req, res) => {
    try {
      const { id } = req.params;
      const messageData = { ...req.body, conversationId: id };
      
      const validation = insertChatMessageSchema.safeParse(messageData);
      if (!validation.success) {
        return res.status(400).json({ message: "Invalid message data" });
      }

      // Save user message
      const userMessage = await storage.createChatMessage(validation.data);

      // Generate AI consensus response
      if (validation.data.role === "user") {
        try {
          // Get current portfolio for context
          const portfolio = await storage.getPortfolioPositions();
          const portfolioContext = portfolio.map(pos => ({
            symbol: pos.stock.symbol,
            name: pos.stock.name,
            shares: pos.shares,
            avgPrice: pos.averagePrice,
            currentValue: pos.currentValue,
            return: pos.returnPercent
          }));

          // Get AI personas for consensus response
          const personas = await storage.getPersonas();
          
          const aiResponse = await openaiService.generateConsensusChat(
            validation.data.content,
            portfolioContext,
            personas
          );

          // Save AI response
          const aiMessage = await storage.createChatMessage({
            conversationId: id,
            role: "assistant",
            content: aiResponse.response,
            metadata: { consensusScore: aiResponse.consensusScore, personas: personas.map(p => p.name) }
          });

          res.json({ userMessage, aiMessage });
        } catch (aiError) {
          console.error("AI response failed:", aiError);
          // Return just user message if AI fails
          res.json({ userMessage, aiMessage: null });
        }
      } else {
        res.json({ userMessage });
      }
    } catch (error) {
      console.error("Chat message error:", error);
      res.status(500).json({ message: "Failed to send message" });
    }
  });

  // Portfolio upload endpoints
  app.post("/api/portfolio/upload", async (req, res) => {
    try {
      const objectStorageService = new ObjectStorageService();
      const uploadURL = await objectStorageService.getObjectEntityUploadURL();
      res.json({ uploadURL });
    } catch (error) {
      res.status(500).json({ message: "Failed to get upload URL" });
    }
  });

  app.post("/api/portfolio/process-upload", async (req, res) => {
    try {
      const { fileName, fileUrl } = req.body;
      
      if (!fileName || !fileUrl) {
        return res.status(400).json({ message: "fileName and fileUrl are required" });
      }

      // Create portfolio upload record
      const upload = await storage.createPortfolioUpload({
        fileName,
        fileUrl,
        status: "PROCESSING"
      });

      // TODO: Process CSV/Excel file and extract positions
      // For now, return the upload record
      res.json(upload);
    } catch (error) {
      res.status(500).json({ message: "Failed to process upload" });
    }
  });

  // Object storage endpoint for serving uploaded files
  app.get("/objects/:objectPath(*)", async (req, res) => {
    const objectStorageService = new ObjectStorageService();
    try {
      const objectFile = await objectStorageService.getObjectEntityFile(req.path);
      objectStorageService.downloadObject(objectFile, res);
    } catch (error) {
      console.error("Error accessing object:", error);
      if (error instanceof ObjectNotFoundError) {
        return res.sendStatus(404);
      }
      return res.sendStatus(500);
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}
