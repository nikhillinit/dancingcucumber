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
  type InsertNewsArticle,
  type ChatConversation,
  type InsertChatConversation,
  type ChatMessage,
  type InsertChatMessage,
  type PortfolioUpload,
  type InsertPortfolioUpload,
  investorPersonas,
  stocks,
  stockAnalyses,
  debates,
  portfolioPositions,
  newsArticles,
  chatConversations,
  chatMessages,
  portfolioUploads
} from "@shared/schema";
import { db } from "./db";
import { eq, desc, sql } from "drizzle-orm";

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

  // Chat
  getChatConversations(): Promise<ChatConversation[]>;
  getChatConversation(id: string): Promise<ChatConversation | undefined>;
  createChatConversation(conversation: InsertChatConversation): Promise<ChatConversation>;
  getChatMessages(conversationId: string): Promise<ChatMessage[]>;
  createChatMessage(message: InsertChatMessage): Promise<ChatMessage>;

  // Portfolio Uploads
  getPortfolioUploads(): Promise<PortfolioUpload[]>;
  getPortfolioUpload(id: string): Promise<PortfolioUpload | undefined>;
  createPortfolioUpload(upload: InsertPortfolioUpload): Promise<PortfolioUpload>;
  updatePortfolioUpload(id: string, updates: Partial<PortfolioUpload>): Promise<PortfolioUpload | undefined>;
}

export class DatabaseStorage implements IStorage {
  constructor() {
    this.initializePersonas();
  }

  private async initializePersonas() {
    try {
      const existingPersonas = await db.select().from(investorPersonas).limit(1);
      if (existingPersonas.length > 0) return;

      const defaultPersonas: InsertInvestorPersona[] = [
        {
          name: "Warren Buffett",
          description: "The Oracle of Omaha - Value investing legend focused on long-term intrinsic value",
          avatar: "🧙‍♂️",
          investmentStyle: "Value",
          personalityTraits: ["patient", "analytical", "conservative", "long-term focused"]
        },
        {
          name: "Cathie Wood",
          description: "Innovation investor focused on disruptive technology and exponential growth",
          avatar: "🚀",
          investmentStyle: "Growth/Innovation", 
          personalityTraits: ["visionary", "risk-taking", "tech-focused", "disruptive"]
        },
        {
          name: "Peter Lynch",
          description: "Growth at a reasonable price (GARP) investor with focus on understandable businesses",
          avatar: "📊",
          investmentStyle: "GARP",
          personalityTraits: ["practical", "research-driven", "opportunistic", "retail-focused"]
        },
        {
          name: "Michael Burry",
          description: "Contrarian value investor known for contrarian bets and deep fundamental analysis",
          avatar: "🕵️",
          investmentStyle: "Deep Value/Contrarian",
          personalityTraits: ["contrarian", "analytical", "skeptical", "independent"]
        },
        {
          name: "Bill Ackman",
          description: "Activist investor focused on high-conviction concentrated positions",
          avatar: "⚡",
          investmentStyle: "Activist/Concentrated",
          personalityTraits: ["activist", "high-conviction", "concentrated", "outspoken"]
        }
      ];

      await db.insert(investorPersonas).values(defaultPersonas);
    } catch (error) {
      console.log("Personas might already exist or database not ready:", error);
    }
  }

  // Personas
  async getPersonas(): Promise<InvestorPersona[]> {
    return await db.select().from(investorPersonas);
  }

  async getPersona(id: string): Promise<InvestorPersona | undefined> {
    const [persona] = await db.select().from(investorPersonas).where(eq(investorPersonas.id, id));
    return persona || undefined;
  }

  async createPersona(persona: InsertInvestorPersona): Promise<InvestorPersona> {
    const [created] = await db.insert(investorPersonas).values(persona).returning();
    return created;
  }

  // Stocks
  async getStocks(): Promise<Stock[]> {
    return await db.select().from(stocks);
  }

  async getStock(id: string): Promise<Stock | undefined> {
    const [stock] = await db.select().from(stocks).where(eq(stocks.id, id));
    return stock || undefined;
  }

  async getStockBySymbol(symbol: string): Promise<Stock | undefined> {
    const [stock] = await db.select().from(stocks).where(eq(stocks.symbol, symbol));
    return stock || undefined;
  }

  async createStock(stock: InsertStock): Promise<Stock> {
    const [created] = await db.insert(stocks).values(stock).returning();
    return created;
  }

  async updateStock(id: string, updates: Partial<Stock>): Promise<Stock | undefined> {
    const [updated] = await db.update(stocks).set(updates).where(eq(stocks.id, id)).returning();
    return updated || undefined;
  }

  // Stock Analyses
  async getAnalysesByStock(stockId: string): Promise<StockAnalysis[]> {
    return await db.select().from(stockAnalyses).where(eq(stockAnalyses.stockId, stockId));
  }

  async getAnalysisByStockAndPersona(stockId: string, personaId: string): Promise<StockAnalysis | undefined> {
    const [analysis] = await db.select().from(stockAnalyses)
      .where(sql`${stockAnalyses.stockId} = ${stockId} AND ${stockAnalyses.personaId} = ${personaId}`);
    return analysis || undefined;
  }

  async createAnalysis(analysis: InsertStockAnalysis): Promise<StockAnalysis> {
    const [created] = await db.insert(stockAnalyses).values(analysis).returning();
    return created;
  }

  async getLatestAnalyses(limit = 10): Promise<(StockAnalysis & { stock: Stock; persona: InvestorPersona })[]> {
    const results = await db.select({
      id: stockAnalyses.id,
      stockId: stockAnalyses.stockId,
      personaId: stockAnalyses.personaId,
      recommendation: stockAnalyses.recommendation,
      confidenceScore: stockAnalyses.confidenceScore,
      reasoning: stockAnalyses.reasoning,
      targetPrice: stockAnalyses.targetPrice,
      analysisDate: stockAnalyses.analysisDate,
      stock: stocks,
      persona: investorPersonas
    })
    .from(stockAnalyses)
    .leftJoin(stocks, eq(stockAnalyses.stockId, stocks.id))
    .leftJoin(investorPersonas, eq(stockAnalyses.personaId, investorPersonas.id))
    .orderBy(desc(stockAnalyses.analysisDate))
    .limit(limit);

    return results.map(result => ({
      ...result,
      stock: result.stock!,
      persona: result.persona!
    }));
  }

  // Debates
  async getDebates(): Promise<Debate[]> {
    return await db.select().from(debates).orderBy(desc(debates.createdAt));
  }

  async getDebate(id: string): Promise<Debate | undefined> {
    const [debate] = await db.select().from(debates).where(eq(debates.id, id));
    return debate || undefined;
  }

  async getDebatesByStock(stockId: string): Promise<Debate[]> {
    return await db.select().from(debates).where(eq(debates.stockId, stockId)).orderBy(desc(debates.createdAt));
  }

  async createDebate(debate: InsertDebate): Promise<Debate> {
    const [created] = await db.insert(debates).values(debate).returning();
    return created;
  }

  async updateDebate(id: string, updates: Partial<Debate>): Promise<Debate | undefined> {
    const [updated] = await db.update(debates).set(updates).where(eq(debates.id, id)).returning();
    return updated || undefined;
  }

  // Portfolio
  async getPortfolioPositions(): Promise<(PortfolioPosition & { stock: Stock })[]> {
    const results = await db.select({
      id: portfolioPositions.id,
      stockId: portfolioPositions.stockId,
      shares: portfolioPositions.shares,
      averagePrice: portfolioPositions.averagePrice,
      currentValue: portfolioPositions.currentValue,
      totalReturn: portfolioPositions.totalReturn,
      returnPercent: portfolioPositions.returnPercent,
      consensusScore: portfolioPositions.consensusScore,
      addedAt: portfolioPositions.addedAt,
      updatedAt: portfolioPositions.updatedAt,
      stock: stocks
    })
    .from(portfolioPositions)
    .leftJoin(stocks, eq(portfolioPositions.stockId, stocks.id))
    .orderBy(desc(portfolioPositions.addedAt));

    return results.map(result => ({
      ...result,
      stock: result.stock!
    }));
  }

  async getPortfolioPosition(id: string): Promise<PortfolioPosition | undefined> {
    const [position] = await db.select().from(portfolioPositions).where(eq(portfolioPositions.id, id));
    return position || undefined;
  }

  async getPortfolioPositionByStock(stockId: string): Promise<PortfolioPosition | undefined> {
    const [position] = await db.select().from(portfolioPositions).where(eq(portfolioPositions.stockId, stockId));
    return position || undefined;
  }

  async createPortfolioPosition(position: InsertPortfolioPosition): Promise<PortfolioPosition> {
    const [created] = await db.insert(portfolioPositions).values(position).returning();
    return created;
  }

  async updatePortfolioPosition(id: string, updates: Partial<PortfolioPosition>): Promise<PortfolioPosition | undefined> {
    const [updated] = await db.update(portfolioPositions).set(updates).where(eq(portfolioPositions.id, id)).returning();
    return updated || undefined;
  }

  async deletePortfolioPosition(id: string): Promise<boolean> {
    const result = await db.delete(portfolioPositions).where(eq(portfolioPositions.id, id));
    return result.rowCount > 0;
  }

  // News
  async getNewsArticles(limit = 20): Promise<NewsArticle[]> {
    return await db.select().from(newsArticles).orderBy(desc(newsArticles.publishedAt)).limit(limit);
  }

  async getNewsArticlesByStock(symbols: string[]): Promise<NewsArticle[]> {
    return await db.select().from(newsArticles)
      .where(sql`${newsArticles.stockSymbols} && ${symbols}`)
      .orderBy(desc(newsArticles.publishedAt));
  }

  async createNewsArticle(article: InsertNewsArticle): Promise<NewsArticle> {
    const [created] = await db.insert(newsArticles).values(article).returning();
    return created;
  }

  // Chat
  async getChatConversations(): Promise<ChatConversation[]> {
    return await db.select().from(chatConversations).orderBy(desc(chatConversations.updatedAt));
  }

  async getChatConversation(id: string): Promise<ChatConversation | undefined> {
    const [conversation] = await db.select().from(chatConversations).where(eq(chatConversations.id, id));
    return conversation || undefined;
  }

  async createChatConversation(conversation: InsertChatConversation): Promise<ChatConversation> {
    const [created] = await db.insert(chatConversations).values(conversation).returning();
    return created;
  }

  async getChatMessages(conversationId: string): Promise<ChatMessage[]> {
    return await db.select().from(chatMessages)
      .where(eq(chatMessages.conversationId, conversationId))
      .orderBy(chatMessages.createdAt);
  }

  async createChatMessage(message: InsertChatMessage): Promise<ChatMessage> {
    const [created] = await db.insert(chatMessages).values(message).returning();
    return created;
  }

  // Portfolio Uploads
  async getPortfolioUploads(): Promise<PortfolioUpload[]> {
    return await db.select().from(portfolioUploads).orderBy(desc(portfolioUploads.uploadedAt));
  }

  async getPortfolioUpload(id: string): Promise<PortfolioUpload | undefined> {
    const [upload] = await db.select().from(portfolioUploads).where(eq(portfolioUploads.id, id));
    return upload || undefined;
  }

  async createPortfolioUpload(upload: InsertPortfolioUpload): Promise<PortfolioUpload> {
    const [created] = await db.insert(portfolioUploads).values(upload).returning();
    return created;
  }

  async updatePortfolioUpload(id: string, updates: Partial<PortfolioUpload>): Promise<PortfolioUpload | undefined> {
    const [updated] = await db.update(portfolioUploads).set(updates).where(eq(portfolioUploads.id, id)).returning();
    return updated || undefined;
  }
}

export const storage = new DatabaseStorage();