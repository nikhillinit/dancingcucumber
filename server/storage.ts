import { 
  type InvestorPersona, 
  type InsertInvestorPersona,
  type Stock,
  type InsertStock,
  type StockAnalysis,
  type InsertStockAnalysis,
  type Debate,
  type InsertDebate,
  type PortfolioPosition,
  type InsertPortfolioPosition,
  type NewsArticle,
  type InsertNewsArticle
} from "@shared/schema";
import { randomUUID } from "crypto";

export interface IStorage {
  // Personas
  getPersonas(): Promise<InvestorPersona[]>;
  getPersona(id: string): Promise<InvestorPersona | undefined>;
  createPersona(persona: InsertInvestorPersona): Promise<InvestorPersona>;

  // Stocks
  getStocks(): Promise<Stock[]>;
  getStock(id: string): Promise<Stock | undefined>;
  getStockBySymbol(symbol: string): Promise<Stock | undefined>;
  createStock(stock: InsertStock): Promise<Stock>;
  updateStock(id: string, updates: Partial<Stock>): Promise<Stock | undefined>;

  // Stock Analyses
  getAnalysesByStock(stockId: string): Promise<StockAnalysis[]>;
  getAnalysisByStockAndPersona(stockId: string, personaId: string): Promise<StockAnalysis | undefined>;
  createAnalysis(analysis: InsertStockAnalysis): Promise<StockAnalysis>;
  getLatestAnalyses(limit?: number): Promise<(StockAnalysis & { stock: Stock; persona: InvestorPersona })[]>;

  // Debates
  getDebates(): Promise<Debate[]>;
  getDebate(id: string): Promise<Debate | undefined>;
  getDebatesByStock(stockId: string): Promise<Debate[]>;
  createDebate(debate: InsertDebate): Promise<Debate>;
  updateDebate(id: string, updates: Partial<Debate>): Promise<Debate | undefined>;

  // Portfolio
  getPortfolioPositions(): Promise<(PortfolioPosition & { stock: Stock })[]>;
  getPortfolioPosition(id: string): Promise<PortfolioPosition | undefined>;
  getPortfolioPositionByStock(stockId: string): Promise<PortfolioPosition | undefined>;
  createPortfolioPosition(position: InsertPortfolioPosition): Promise<PortfolioPosition>;
  updatePortfolioPosition(id: string, updates: Partial<PortfolioPosition>): Promise<PortfolioPosition | undefined>;
  deletePortfolioPosition(id: string): Promise<boolean>;

  // News
  getNewsArticles(limit?: number): Promise<NewsArticle[]>;
  getNewsArticlesByStock(symbols: string[]): Promise<NewsArticle[]>;
  createNewsArticle(article: InsertNewsArticle): Promise<NewsArticle>;
}

export class MemStorage implements IStorage {
  private personas: Map<string, InvestorPersona>;
  private stocks: Map<string, Stock>;
  private stockAnalyses: Map<string, StockAnalysis>;
  private debates: Map<string, Debate>;
  private portfolioPositions: Map<string, PortfolioPosition>;
  private newsArticles: Map<string, NewsArticle>;

  constructor() {
    this.personas = new Map();
    this.stocks = new Map();
    this.stockAnalyses = new Map();
    this.debates = new Map();
    this.portfolioPositions = new Map();
    this.newsArticles = new Map();

    // Initialize with default personas
    this.initializePersonas();
  }

  private initializePersonas() {
    const defaultPersonas: InsertInvestorPersona[] = [
      {
        name: "Warren Buffett",
        description: "The Oracle of Omaha, seeks wonderful companies at a fair price",
        avatar: "warren-buffett",
        investmentStyle: "VALUE",
        personalityTraits: ["patient", "long-term focused", "fundamentals-driven", "moat-focused"]
      },
      {
        name: "Cathie Wood",
        description: "The queen of growth investing, believes in innovation and disruption",
        avatar: "cathie-wood",
        investmentStyle: "GROWTH",
        personalityTraits: ["innovative", "disruptive", "tech-focused", "forward-thinking"]
      },
      {
        name: "Peter Lynch",
        description: "Practical investor who seeks ten-baggers in everyday businesses",
        avatar: "peter-lynch",
        investmentStyle: "GROWTH_AT_REASONABLE_PRICE",
        personalityTraits: ["practical", "research-driven", "consumer-focused", "opportunistic"]
      },
      {
        name: "Michael Burry",
        description: "The Big Short contrarian who hunts for deep value",
        avatar: "michael-burry",
        investmentStyle: "CONTRARIAN_VALUE",
        personalityTraits: ["contrarian", "analytical", "skeptical", "independent"]
      },
      {
        name: "Bill Ackman",
        description: "Activist investor who takes bold positions and pushes for change",
        avatar: "bill-ackman",
        investmentStyle: "ACTIVIST",
        personalityTraits: ["bold", "activist", "concentrated", "change-oriented"]
      }
    ];

    defaultPersonas.forEach(persona => {
      const id = randomUUID();
      this.personas.set(id, { ...persona, id, avatar: persona.avatar || null });
    });
  }

  // Personas
  async getPersonas(): Promise<InvestorPersona[]> {
    return Array.from(this.personas.values());
  }

  async getPersona(id: string): Promise<InvestorPersona | undefined> {
    return this.personas.get(id);
  }

  async createPersona(persona: InsertInvestorPersona): Promise<InvestorPersona> {
    const id = randomUUID();
    const newPersona: InvestorPersona = { ...persona, id, avatar: persona.avatar || null };
    this.personas.set(id, newPersona);
    return newPersona;
  }

  // Stocks
  async getStocks(): Promise<Stock[]> {
    return Array.from(this.stocks.values());
  }

  async getStock(id: string): Promise<Stock | undefined> {
    return this.stocks.get(id);
  }

  async getStockBySymbol(symbol: string): Promise<Stock | undefined> {
    return Array.from(this.stocks.values()).find(stock => stock.symbol === symbol);
  }

  async createStock(stock: InsertStock): Promise<Stock> {
    const id = randomUUID();
    const newStock: Stock = { 
      ...stock, 
      id,
      currentPrice: stock.currentPrice || null,
      priceChange: stock.priceChange || null,
      priceChangePercent: stock.priceChangePercent || null,
      marketData: stock.marketData || null,
      lastUpdated: new Date()
    };
    this.stocks.set(id, newStock);
    return newStock;
  }

  async updateStock(id: string, updates: Partial<Stock>): Promise<Stock | undefined> {
    const stock = this.stocks.get(id);
    if (!stock) return undefined;

    const updatedStock = { 
      ...stock, 
      ...updates,
      lastUpdated: new Date()
    };
    this.stocks.set(id, updatedStock);
    return updatedStock;
  }

  // Stock Analyses
  async getAnalysesByStock(stockId: string): Promise<StockAnalysis[]> {
    return Array.from(this.stockAnalyses.values()).filter(analysis => analysis.stockId === stockId);
  }

  async getAnalysisByStockAndPersona(stockId: string, personaId: string): Promise<StockAnalysis | undefined> {
    return Array.from(this.stockAnalyses.values())
      .find(analysis => analysis.stockId === stockId && analysis.personaId === personaId);
  }

  async createAnalysis(analysis: InsertStockAnalysis): Promise<StockAnalysis> {
    const id = randomUUID();
    const newAnalysis: StockAnalysis = { 
      ...analysis, 
      id,
      targetPrice: analysis.targetPrice || null,
      analysisDate: new Date()
    };
    this.stockAnalyses.set(id, newAnalysis);
    return newAnalysis;
  }

  async getLatestAnalyses(limit = 10): Promise<(StockAnalysis & { stock: Stock; persona: InvestorPersona })[]> {
    const analyses = Array.from(this.stockAnalyses.values())
      .sort((a, b) => (b.analysisDate?.getTime() || 0) - (a.analysisDate?.getTime() || 0))
      .slice(0, limit);

    const enrichedAnalyses = analyses.map(analysis => {
      const stock = this.stocks.get(analysis.stockId);
      const persona = this.personas.get(analysis.personaId);
      return {
        ...analysis,
        stock: stock!,
        persona: persona!
      };
    }).filter(analysis => analysis.stock && analysis.persona);

    return enrichedAnalyses;
  }

  // Debates
  async getDebates(): Promise<Debate[]> {
    return Array.from(this.debates.values());
  }

  async getDebate(id: string): Promise<Debate | undefined> {
    return this.debates.get(id);
  }

  async getDebatesByStock(stockId: string): Promise<Debate[]> {
    return Array.from(this.debates.values()).filter(debate => debate.stockId === stockId);
  }

  async createDebate(debate: InsertDebate): Promise<Debate> {
    const id = randomUUID();
    const now = new Date();
    const newDebate: Debate = { 
      ...debate, 
      id,
      consensusScore: debate.consensusScore || null,
      createdAt: now,
      updatedAt: now
    };
    this.debates.set(id, newDebate);
    return newDebate;
  }

  async updateDebate(id: string, updates: Partial<Debate>): Promise<Debate | undefined> {
    const debate = this.debates.get(id);
    if (!debate) return undefined;

    const updatedDebate = { 
      ...debate, 
      ...updates,
      updatedAt: new Date()
    };
    this.debates.set(id, updatedDebate);
    return updatedDebate;
  }

  // Portfolio
  async getPortfolioPositions(): Promise<(PortfolioPosition & { stock: Stock })[]> {
    const positions = Array.from(this.portfolioPositions.values());
    const enrichedPositions = positions.map(position => {
      const stock = this.stocks.get(position.stockId);
      return {
        ...position,
        stock: stock!
      };
    }).filter(position => position.stock);

    return enrichedPositions;
  }

  async getPortfolioPosition(id: string): Promise<PortfolioPosition | undefined> {
    return this.portfolioPositions.get(id);
  }

  async getPortfolioPositionByStock(stockId: string): Promise<PortfolioPosition | undefined> {
    return Array.from(this.portfolioPositions.values())
      .find(position => position.stockId === stockId);
  }

  async createPortfolioPosition(position: InsertPortfolioPosition): Promise<PortfolioPosition> {
    const id = randomUUID();
    const now = new Date();
    const newPosition: PortfolioPosition = { 
      ...position, 
      id,
      currentValue: position.currentValue || null,
      totalReturn: position.totalReturn || null,
      returnPercent: position.returnPercent || null,
      consensusScore: position.consensusScore || null,
      addedAt: now,
      updatedAt: now
    };
    this.portfolioPositions.set(id, newPosition);
    return newPosition;
  }

  async updatePortfolioPosition(id: string, updates: Partial<PortfolioPosition>): Promise<PortfolioPosition | undefined> {
    const position = this.portfolioPositions.get(id);
    if (!position) return undefined;

    const updatedPosition = { 
      ...position, 
      ...updates,
      updatedAt: new Date()
    };
    this.portfolioPositions.set(id, updatedPosition);
    return updatedPosition;
  }

  async deletePortfolioPosition(id: string): Promise<boolean> {
    return this.portfolioPositions.delete(id);
  }

  // News
  async getNewsArticles(limit = 20): Promise<NewsArticle[]> {
    return Array.from(this.newsArticles.values())
      .sort((a, b) => b.publishedAt.getTime() - a.publishedAt.getTime())
      .slice(0, limit);
  }

  async getNewsArticlesByStock(symbols: string[]): Promise<NewsArticle[]> {
    return Array.from(this.newsArticles.values())
      .filter(article => 
        article.stockSymbols?.some(symbol => 
          symbols.includes(symbol)
        )
      )
      .sort((a, b) => b.publishedAt.getTime() - a.publishedAt.getTime());
  }

  async createNewsArticle(article: InsertNewsArticle): Promise<NewsArticle> {
    const id = randomUUID();
    const newArticle: NewsArticle = { 
      ...article, 
      id,
      description: article.description || null,
      imageUrl: article.imageUrl || null,
      stockSymbols: article.stockSymbols || null,
      sentiment: article.sentiment || null,
      impact: article.impact || null
    };
    this.newsArticles.set(id, newArticle);
    return newArticle;
  }
}

export const storage = new MemStorage();
