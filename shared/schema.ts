import { sql } from "drizzle-orm";
import { pgTable, text, varchar, decimal, timestamp, jsonb, integer } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const investorPersonas = pgTable("investor_personas", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  name: text("name").notNull(),
  description: text("description").notNull(),
  avatar: text("avatar"),
  investmentStyle: text("investment_style").notNull(),
  personalityTraits: jsonb("personality_traits").$type<string[]>().notNull(),
});

export const stocks = pgTable("stocks", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  symbol: text("symbol").notNull().unique(),
  name: text("name").notNull(),
  currentPrice: decimal("current_price", { precision: 10, scale: 2 }),
  priceChange: decimal("price_change", { precision: 10, scale: 2 }),
  priceChangePercent: decimal("price_change_percent", { precision: 5, scale: 2 }),
  marketData: jsonb("market_data").$type<Record<string, any>>(),
  lastUpdated: timestamp("last_updated").defaultNow(),
});

export const stockAnalyses = pgTable("stock_analyses", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  stockId: varchar("stock_id").notNull().references(() => stocks.id),
  personaId: varchar("persona_id").notNull().references(() => investorPersonas.id),
  recommendation: text("recommendation").notNull(), // "BUY", "HOLD", "SELL", "STRONG_BUY", "STRONG_SELL"
  confidenceScore: integer("confidence_score").notNull(), // 0-100
  reasoning: text("reasoning").notNull(),
  targetPrice: decimal("target_price", { precision: 10, scale: 2 }),
  analysisDate: timestamp("analysis_date").defaultNow(),
});

export const debates = pgTable("debates", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  stockId: varchar("stock_id").notNull().references(() => stocks.id),
  title: text("title").notNull(),
  status: text("status").notNull(), // "ACTIVE", "CLOSED"
  participants: jsonb("participants").$type<string[]>().notNull(), // persona IDs
  messages: jsonb("messages").$type<Array<{
    personaId: string;
    message: string;
    timestamp: string;
    type: "argument" | "counter_argument" | "consensus";
  }>>().notNull(),
  consensusScore: integer("consensus_score"), // 0-100
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const portfolioPositions = pgTable("portfolio_positions", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  stockId: varchar("stock_id").notNull().references(() => stocks.id),
  shares: decimal("shares", { precision: 10, scale: 4 }).notNull(),
  averagePrice: decimal("average_price", { precision: 10, scale: 2 }).notNull(),
  currentValue: decimal("current_value", { precision: 12, scale: 2 }),
  totalReturn: decimal("total_return", { precision: 12, scale: 2 }),
  returnPercent: decimal("return_percent", { precision: 5, scale: 2 }),
  consensusScore: integer("consensus_score"), // 0-100
  addedAt: timestamp("added_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const newsArticles = pgTable("news_articles", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  title: text("title").notNull(),
  description: text("description"),
  url: text("url").notNull(),
  imageUrl: text("image_url"),
  source: text("source").notNull(),
  publishedAt: timestamp("published_at").notNull(),
  stockSymbols: jsonb("stock_symbols").$type<string[]>(),
  sentiment: text("sentiment"), // "BULLISH", "BEARISH", "NEUTRAL"
  impact: text("impact"), // "HIGH", "MEDIUM", "LOW"
});

// Insert schemas
export const insertInvestorPersonaSchema = createInsertSchema(investorPersonas).omit({
  id: true,
});

export const insertStockSchema = createInsertSchema(stocks).omit({
  id: true,
  lastUpdated: true,
});

export const insertStockAnalysisSchema = createInsertSchema(stockAnalyses).omit({
  id: true,
  analysisDate: true,
});

export const insertDebateSchema = createInsertSchema(debates).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const insertPortfolioPositionSchema = createInsertSchema(portfolioPositions).omit({
  id: true,
  addedAt: true,
  updatedAt: true,
});

export const insertNewsArticleSchema = createInsertSchema(newsArticles).omit({
  id: true,
});

// Types
export type InvestorPersona = typeof investorPersonas.$inferSelect;
export type InsertInvestorPersona = z.infer<typeof insertInvestorPersonaSchema>;

export type Stock = typeof stocks.$inferSelect;
export type InsertStock = z.infer<typeof insertStockSchema>;

export type StockAnalysis = typeof stockAnalyses.$inferSelect;
export type InsertStockAnalysis = z.infer<typeof insertStockAnalysisSchema>;

export type Debate = typeof debates.$inferSelect;
export type InsertDebate = z.infer<typeof insertDebateSchema>;

export type PortfolioPosition = typeof portfolioPositions.$inferSelect;
export type InsertPortfolioPosition = z.infer<typeof insertPortfolioPositionSchema>;

export type NewsArticle = typeof newsArticles.$inferSelect;
export type InsertNewsArticle = z.infer<typeof insertNewsArticleSchema>;
