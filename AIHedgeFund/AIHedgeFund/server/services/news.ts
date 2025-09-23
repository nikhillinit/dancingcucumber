interface NewsSource {
  name: string;
  url: string;
}

interface NewsAPIArticle {
  title: string;
  description: string;
  url: string;
  urlToImage: string;
  publishedAt: string;
  source: NewsSource;
}

export class NewsService {
  private apiKey: string;

  constructor() {
    this.apiKey = process.env.NEWS_API_KEY || "";
  }

  async getMarketNews(limit: number = 10): Promise<NewsAPIArticle[]> {
    try {
      if (!this.apiKey) {
        return this.getMockNews();
      }

      const response = await fetch(
        `https://newsapi.org/v2/everything?q=stock market OR finance OR investing&sortBy=publishedAt&pageSize=${limit}&apiKey=${this.apiKey}`
      );

      if (!response.ok) {
        throw new Error(`Failed to fetch news: ${response.statusText}`);
      }

      const data = await response.json();
      return data.articles || [];
    } catch (error) {
      console.error('Error fetching market news:', error);
      return this.getMockNews();
    }
  }

  async getStockNews(symbol: string, limit: number = 5): Promise<NewsAPIArticle[]> {
    try {
      if (!this.apiKey) {
        return this.getMockStockNews(symbol);
      }

      const response = await fetch(
        `https://newsapi.org/v2/everything?q=${symbol}&sortBy=publishedAt&pageSize=${limit}&apiKey=${this.apiKey}`
      );

      if (!response.ok) {
        throw new Error(`Failed to fetch stock news: ${response.statusText}`);
      }

      const data = await response.json();
      return data.articles || [];
    } catch (error) {
      console.error(`Error fetching news for ${symbol}:`, error);
      return this.getMockStockNews(symbol);
    }
  }

  private getMockNews(): NewsAPIArticle[] {
    return [
      {
        title: "Fed Signals Potential Rate Cut as Inflation Moderates",
        description: "Markets rally on dovish Fed commentary, with tech stocks leading gains as investors anticipate lower borrowing costs.",
        url: "https://example.com/fed-rate-cut",
        urlToImage: "https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?w=400&h=300&fit=crop",
        publishedAt: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(), // 2 hours ago
        source: { name: "CNBC", url: "https://cnbc.com" }
      },
      {
        title: "Apple Unveils New AI Features for iOS 18",
        description: "Enhanced Siri capabilities and machine learning integration drive stock higher as AI becomes central to Apple's strategy.",
        url: "https://example.com/apple-ai-features",
        urlToImage: "https://images.unsplash.com/photo-1592179900008-3d5c6276f3b8?w=400&h=300&fit=crop",
        publishedAt: new Date(Date.now() - 4 * 60 * 60 * 1000).toISOString(), // 4 hours ago
        source: { name: "TechCrunch", url: "https://techcrunch.com" }
      },
      {
        title: "NVIDIA Reports Record Data Center Revenue",
        description: "AI chip demand continues to surge as NVIDIA beats earnings expectations with 150% year-over-year growth in data center business.",
        url: "https://example.com/nvidia-earnings",
        urlToImage: "https://images.unsplash.com/photo-1518709268805-4e9042af2176?w=400&h=300&fit=crop",
        publishedAt: new Date(Date.now() - 6 * 60 * 60 * 1000).toISOString(), // 6 hours ago
        source: { name: "Reuters", url: "https://reuters.com" }
      },
      {
        title: "Tesla Stock Slides on Production Concerns",
        description: "Shares decline after reports of production delays at Gigafactory Texas, raising questions about Q4 delivery targets.",
        url: "https://example.com/tesla-production",
        urlToImage: "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=400&h=300&fit=crop",
        publishedAt: new Date(Date.now() - 8 * 60 * 60 * 1000).toISOString(), // 8 hours ago
        source: { name: "Bloomberg", url: "https://bloomberg.com" }
      },
      {
        title: "Microsoft Azure Revenue Growth Accelerates",
        description: "Cloud computing division shows strong momentum with 35% growth, driven by AI and enterprise digital transformation.",
        url: "https://example.com/microsoft-azure",
        urlToImage: "https://images.unsplash.com/photo-1633356122544-f134324a6cee?w=400&h=300&fit=crop",
        publishedAt: new Date(Date.now() - 12 * 60 * 60 * 1000).toISOString(), // 12 hours ago
        source: { name: "Wall Street Journal", url: "https://wsj.com" }
      }
    ];
  }

  private getMockStockNews(symbol: string): NewsAPIArticle[] {
    const stockNews: Record<string, NewsAPIArticle[]> = {
      'AAPL': [
        {
          title: "Apple iPhone 15 Sales Exceed Expectations in China",
          description: "Strong demand for new iPhone models in key market boosts investor confidence ahead of holiday season.",
          url: "https://example.com/apple-china-sales",
          urlToImage: "https://images.unsplash.com/photo-1592179900008-3d5c6276f3b8?w=400&h=300&fit=crop",
          publishedAt: new Date(Date.now() - 3 * 60 * 60 * 1000).toISOString(),
          source: { name: "Financial Times", url: "https://ft.com" }
        }
      ],
      'NVDA': [
        {
          title: "NVIDIA Partners with Major Automakers for AI Chips",
          description: "New partnerships expand NVIDIA's presence in autonomous vehicle market, diversifying beyond data center business.",
          url: "https://example.com/nvidia-auto-partnerships",
          urlToImage: "https://images.unsplash.com/photo-1518709268805-4e9042af2176?w=400&h=300&fit=crop",
          publishedAt: new Date(Date.now() - 5 * 60 * 60 * 1000).toISOString(),
          source: { name: "TechCrunch", url: "https://techcrunch.com" }
        }
      ]
    };

    return stockNews[symbol.toUpperCase()] || this.getMockNews().slice(0, 2);
  }

  async analyzeNewsSentiment(articles: NewsAPIArticle[]): Promise<{
    overallSentiment: "BULLISH" | "BEARISH" | "NEUTRAL";
    sentimentScore: number; // -1 to 1
    keyTopics: string[];
  }> {
    // Simple sentiment analysis based on keywords
    const bullishKeywords = ['growth', 'profit', 'beat', 'exceed', 'strong', 'gain', 'rise', 'up', 'positive'];
    const bearishKeywords = ['decline', 'loss', 'miss', 'weak', 'fall', 'drop', 'down', 'negative', 'concern'];

    let sentimentScore = 0;
    const topics = new Set<string>();

    articles.forEach(article => {
      const text = `${article.title} ${article.description}`.toLowerCase();
      
      bullishKeywords.forEach(keyword => {
        if (text.includes(keyword)) sentimentScore += 0.1;
      });
      
      bearishKeywords.forEach(keyword => {
        if (text.includes(keyword)) sentimentScore -= 0.1;
      });

      // Extract simple topics
      if (text.includes('ai') || text.includes('artificial intelligence')) topics.add('AI');
      if (text.includes('earnings') || text.includes('revenue')) topics.add('Earnings');
      if (text.includes('fed') || text.includes('interest rate')) topics.add('Fed Policy');
      if (text.includes('china') || text.includes('trade')) topics.add('International');
    });

    const normalizedScore = Math.max(-1, Math.min(1, sentimentScore));
    let overallSentiment: "BULLISH" | "BEARISH" | "NEUTRAL";

    if (normalizedScore > 0.2) {
      overallSentiment = "BULLISH";
    } else if (normalizedScore < -0.2) {
      overallSentiment = "BEARISH";
    } else {
      overallSentiment = "NEUTRAL";
    }

    return {
      overallSentiment,
      sentimentScore: normalizedScore,
      keyTopics: Array.from(topics)
    };
  }
}

export const newsService = new NewsService();
