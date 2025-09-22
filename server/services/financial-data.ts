interface StockQuote {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  marketCap?: number;
  volume?: number;
  pe?: number;
  eps?: number;
  high52Week?: number;
  low52Week?: number;
}

interface FinancialMetrics {
  revenue: number;
  netIncome: number;
  grossMargin: number;
  operatingMargin: number;
  returnOnEquity: number;
  debtToEquity: number;
  currentRatio: number;
  bookValue: number;
}

export class FinancialDataService {
  private apiKey: string;

  constructor() {
    this.apiKey = process.env.FINANCIAL_DATASETS_API_KEY || process.env.ALPHA_VANTAGE_API_KEY || "";
  }

  async getStockQuote(symbol: string): Promise<StockQuote> {
    try {
      // For demo purposes with major stocks, return mock data if no API key
      if (!this.apiKey && this.isMajorStock(symbol)) {
        return this.getMockStockData(symbol);
      }

      if (!this.apiKey) {
        throw new Error("Financial data API key not configured");
      }

      // Use Alpha Vantage API as fallback
      const response = await fetch(
        `https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=${symbol}&apikey=${this.apiKey}`
      );

      if (!response.ok) {
        throw new Error(`Failed to fetch stock data: ${response.statusText}`);
      }

      const data = await response.json();
      const quote = data["Global Quote"];

      if (!quote) {
        throw new Error("Invalid response from financial data API");
      }

      return {
        symbol: quote["01. symbol"],
        name: symbol, // API doesn't provide company name in this endpoint
        price: parseFloat(quote["05. price"]),
        change: parseFloat(quote["09. change"]),
        changePercent: parseFloat(quote["10. change percent"].replace('%', '')),
        volume: parseInt(quote["06. volume"]),
        high52Week: parseFloat(quote["03. high"]),
        low52Week: parseFloat(quote["04. low"])
      };
    } catch (error) {
      console.error(`Error fetching stock quote for ${symbol}:`, error);
      
      // Fallback to mock data for major stocks
      if (this.isMajorStock(symbol)) {
        return this.getMockStockData(symbol);
      }
      
      throw error;
    }
  }

  async getFinancialMetrics(symbol: string): Promise<FinancialMetrics> {
    try {
      if (!this.apiKey) {
        throw new Error("Financial data API key not configured");
      }

      // This would use a more comprehensive financial data API
      // For now, return mock data or throw error
      if (this.isMajorStock(symbol)) {
        return this.getMockFinancialMetrics(symbol);
      }

      throw new Error("Financial metrics not available for this stock");
    } catch (error) {
      console.error(`Error fetching financial metrics for ${symbol}:`, error);
      throw error;
    }
  }

  private isMajorStock(symbol: string): boolean {
    const majorStocks = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'META'];
    return majorStocks.includes(symbol.toUpperCase());
  }

  private getMockStockData(symbol: string): StockQuote {
    const mockData: Record<string, Partial<StockQuote>> = {
      'AAPL': {
        name: 'Apple Inc.',
        price: 175.43,
        change: 3.67,
        changePercent: 2.14,
        marketCap: 2800000000000,
        volume: 45000000,
        pe: 28.5,
        eps: 6.15
      },
      'NVDA': {
        name: 'NVIDIA Corporation',
        price: 485.20,
        change: 22.15,
        changePercent: 4.78,
        marketCap: 1200000000000,
        volume: 35000000,
        pe: 65.2,
        eps: 7.43
      },
      'TSLA': {
        name: 'Tesla, Inc.',
        price: 248.87,
        change: -5.23,
        changePercent: -2.06,
        marketCap: 780000000000,
        volume: 28000000,
        pe: 85.4,
        eps: 2.91
      },
      'GOOGL': {
        name: 'Alphabet Inc.',
        price: 138.21,
        change: 1.89,
        changePercent: 1.39,
        marketCap: 1750000000000,
        volume: 22000000,
        pe: 25.8,
        eps: 5.35
      },
      'MSFT': {
        name: 'Microsoft Corporation',
        price: 378.85,
        change: 2.45,
        changePercent: 0.65,
        marketCap: 2810000000000,
        volume: 18000000,
        pe: 32.1,
        eps: 11.79
      }
    };

    const base = mockData[symbol.toUpperCase()] || {
      name: `${symbol} Corp`,
      price: 100 + Math.random() * 200,
      change: (Math.random() - 0.5) * 20,
      changePercent: (Math.random() - 0.5) * 10
    };

    return {
      symbol: symbol.toUpperCase(),
      ...base,
      high52Week: base.price! * (1 + Math.random() * 0.5),
      low52Week: base.price! * (1 - Math.random() * 0.3),
      volume: base.volume || Math.floor(Math.random() * 50000000),
      marketCap: base.marketCap || base.price! * 1000000000,
      pe: base.pe || 15 + Math.random() * 50,
      eps: base.eps || base.price! / (15 + Math.random() * 50)
    } as StockQuote;
  }

  private getMockFinancialMetrics(symbol: string): FinancialMetrics {
    return {
      revenue: 300000000000 + Math.random() * 100000000000,
      netIncome: 50000000000 + Math.random() * 30000000000,
      grossMargin: 0.35 + Math.random() * 0.3,
      operatingMargin: 0.2 + Math.random() * 0.2,
      returnOnEquity: 0.15 + Math.random() * 0.2,
      debtToEquity: Math.random() * 2,
      currentRatio: 1 + Math.random() * 2,
      bookValue: 20 + Math.random() * 100
    };
  }

  async searchStocks(query: string): Promise<StockQuote[]> {
    try {
      // For demo, return filtered major stocks
      const majorStocks = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'META'];
      const filteredSymbols = majorStocks.filter(symbol => 
        symbol.includes(query.toUpperCase()) || 
        this.getMockStockData(symbol).name.toLowerCase().includes(query.toLowerCase())
      );

      const results = await Promise.all(
        filteredSymbols.map(symbol => this.getStockQuote(symbol))
      );

      return results;
    } catch (error) {
      console.error('Error searching stocks:', error);
      return [];
    }
  }
}

export const financialDataService = new FinancialDataService();
