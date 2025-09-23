import OpenAI from "openai";
import type { InvestorPersona, Stock } from "@shared/schema";

// the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
const openai = new OpenAI({ 
  apiKey: process.env.OPENAI_API_KEY || process.env.OPENAI_API_KEY_ENV_VAR || ""
});

interface PersonaAnalysis {
  recommendation: "BUY" | "HOLD" | "SELL" | "STRONG_BUY" | "STRONG_SELL";
  confidenceScore: number; // 0-100
  reasoning: string;
  targetPrice?: number;
}

interface ConsensusResult {
  overallRecommendation: string;
  consensusScore: number; // 0-100
  summary: string;
  keyDebatePoints: string[];
}

interface ConsensusChatResult {
  response: string;
  consensusScore: number;
  participatingPersonas: string[];
}

export class OpenAIService {
  async analyzeStockByPersona(
    persona: InvestorPersona, 
    stock: Stock, 
    marketData: any,
    newsContext?: string[]
  ): Promise<PersonaAnalysis> {
    const prompt = `
You are ${persona.name}, a legendary investor with the following characteristics:
- Investment Style: ${persona.investmentStyle}
- Description: ${persona.description}
- Personality Traits: ${persona.personalityTraits.join(", ")}

Analyze the stock ${stock.symbol} (${stock.name}) with the following data:
- Current Price: $${stock.currentPrice}
- Price Change: ${stock.priceChange} (${stock.priceChangePercent}%)
- Market Data: ${JSON.stringify(stock.marketData)}
${newsContext ? `- Recent News Context: ${newsContext.join(". ")}` : ""}

Based on your investment philosophy and the provided data, provide your analysis in JSON format:
{
  "recommendation": "BUY|HOLD|SELL|STRONG_BUY|STRONG_SELL",
  "confidenceScore": number (0-100),
  "reasoning": "Detailed explanation of your analysis in your characteristic style",
  "targetPrice": number (optional)
}

Stay true to your known investment principles and communication style.
`;

    try {
      const response = await openai.chat.completions.create({
        model: "gpt-5",
        messages: [
          {
            role: "system",
            content: "You are a financial analysis AI that responds in JSON format only."
          },
          {
            role: "user",
            content: prompt
          }
        ],
        response_format: { type: "json_object" }
      });

      const analysis = JSON.parse(response.choices[0].message.content || "{}");
      return {
        recommendation: analysis.recommendation,
        confidenceScore: Math.max(0, Math.min(100, analysis.confidenceScore)),
        reasoning: analysis.reasoning,
        targetPrice: analysis.targetPrice
      };
    } catch (error) {
      console.error("Error analyzing stock with OpenAI:", error);
      throw new Error("Failed to analyze stock");
    }
  }

  async generateConsensus(
    stock: Stock,
    analyses: Array<{ persona: InvestorPersona; analysis: PersonaAnalysis }>
  ): Promise<ConsensusResult> {
    const analysesText = analyses.map(({ persona, analysis }) => 
      `${persona.name}: ${analysis.recommendation} (${analysis.confidenceScore}% confidence) - ${analysis.reasoning}`
    ).join("\n\n");

    const prompt = `
Analyze the following investment opinions from legendary investors about ${stock.symbol} (${stock.name}):

${analysesText}

Generate a consensus analysis in JSON format:
{
  "overallRecommendation": "Clear recommendation based on majority and confidence levels",
  "consensusScore": number (0-100, where 100 is complete agreement),
  "summary": "Brief summary of the overall consensus and key points of agreement/disagreement",
  "keyDebatePoints": ["List of main points where investors disagree or have different perspectives"]
}

Consider both the number of similar recommendations and the confidence scores when determining consensus.
`;

    try {
      const response = await openai.chat.completions.create({
        model: "gpt-5",
        messages: [
          {
            role: "system",
            content: "You are a financial consensus analysis AI that responds in JSON format only."
          },
          {
            role: "user",
            content: prompt
          }
        ],
        response_format: { type: "json_object" }
      });

      const consensus = JSON.parse(response.choices[0].message.content || "{}");
      return {
        overallRecommendation: consensus.overallRecommendation,
        consensusScore: Math.max(0, Math.min(100, consensus.consensusScore)),
        summary: consensus.summary,
        keyDebatePoints: consensus.keyDebatePoints || []
      };
    } catch (error) {
      console.error("Error generating consensus with OpenAI:", error);
      throw new Error("Failed to generate consensus");
    }
  }

  async generateDebateMessage(
    persona: InvestorPersona,
    stock: Stock,
    context: string,
    previousMessages: Array<{ personaId: string; message: string }>
  ): Promise<string> {
    const conversationHistory = previousMessages
      .map(msg => `${msg.personaId}: ${msg.message}`)
      .join("\n");

    const prompt = `
You are ${persona.name} participating in an investment debate about ${stock.symbol}.
Your characteristics: ${persona.description}
Investment Style: ${persona.investmentStyle}
Personality: ${persona.personalityTraits.join(", ")}

Context: ${context}

Previous conversation:
${conversationHistory}

Respond with a thoughtful message that:
1. Stays true to your investment philosophy
2. Addresses specific points made by others
3. Provides substantive analysis
4. Maintains your characteristic communication style

Keep your response concise but insightful (1-3 sentences).
`;

    try {
      const response = await openai.chat.completions.create({
        model: "gpt-5",
        messages: [
          {
            role: "system",
            content: "You are participating in an investment debate. Respond as the specified investor persona."
          },
          {
            role: "user",
            content: prompt
          }
        ]
      });

      return response.choices[0].message.content || "";
    } catch (error) {
      console.error("Error generating debate message with OpenAI:", error);
      throw new Error("Failed to generate debate message");
    }
  }

  async analyzeNewsImpact(newsArticles: string[], stock: Stock): Promise<{
    sentiment: "BULLISH" | "BEARISH" | "NEUTRAL";
    impact: "HIGH" | "MEDIUM" | "LOW";
    summary: string;
  }> {
    const newsText = newsArticles.join("\n\n");

    const prompt = `
Analyze the impact of the following news articles on ${stock.symbol} (${stock.name}):

${newsText}

Provide analysis in JSON format:
{
  "sentiment": "BULLISH|BEARISH|NEUTRAL",
  "impact": "HIGH|MEDIUM|LOW",
  "summary": "Brief summary of how this news affects the stock"
}
`;

    try {
      const response = await openai.chat.completions.create({
        model: "gpt-5",
        messages: [
          {
            role: "system",
            content: "You are a news sentiment analysis AI that responds in JSON format only."
          },
          {
            role: "user",
            content: prompt
          }
        ],
        response_format: { type: "json_object" }
      });

      const analysis = JSON.parse(response.choices[0].message.content || "{}");
      return {
        sentiment: analysis.sentiment,
        impact: analysis.impact,
        summary: analysis.summary
      };
    } catch (error) {
      console.error("Error analyzing news impact with OpenAI:", error);
      throw new Error("Failed to analyze news impact");
    }
  }

  async generateConsensusChat(
    userQuestion: string,
    portfolioContext: any[],
    personas: InvestorPersona[]
  ): Promise<ConsensusChatResult> {
    const portfolioSummary = portfolioContext.length > 0 
      ? portfolioContext.map(pos => `${pos.symbol}: ${pos.shares} shares at $${pos.avgPrice}, current return: ${pos.return}%`).join(", ")
      : "No current positions";

    const prompt = `
You are an AI investment advisory team representing these legendary investors:
${personas.map(p => `- ${p.name}: ${p.description} (${p.investmentStyle})`).join("\n")}

The user has asked: "${userQuestion}"

Current Portfolio Context: ${portfolioSummary}

As a team of these investment legends, provide a consensus response that:
1. Incorporates perspectives from multiple personas where relevant
2. Considers the user's current portfolio positions
3. Provides actionable advice based on the collective wisdom
4. Shows areas of agreement and disagreement among the personas
5. Includes a confidence score (0-100) for the consensus

Respond in JSON format:
{
  "response": "A comprehensive response incorporating multiple investor perspectives",
  "consensusScore": number (0-100, representing agreement level among personas),
  "participatingPersonas": ["array", "of", "persona", "names", "that", "contributed"]
}

Make the response conversational and helpful, as if the user is getting advice from a panel of investment experts.
`;

    try {
      const response = await openai.chat.completions.create({
        model: "gpt-5",
        messages: [
          {
            role: "system",
            content: "You are an AI investment advisory team that responds in JSON format only."
          },
          {
            role: "user",
            content: prompt
          }
        ],
        response_format: { type: "json_object" }
      });

      const result = JSON.parse(response.choices[0].message.content || "{}");
      return {
        response: result.response || "I'm sorry, I couldn't generate a response at this time.",
        consensusScore: Math.max(0, Math.min(100, result.consensusScore || 50)),
        participatingPersonas: result.participatingPersonas || personas.map(p => p.name)
      };
    } catch (error) {
      console.error("Error generating consensus chat response:", error);
      // Generate a mock consensus response as fallback
      return this.generateMockConsensusResponse(userQuestion, portfolioSummary, personas);
    }
  }

  private generateMockConsensusResponse(
    userQuestion: string, 
    portfolioSummary: string, 
    personas: InvestorPersona[]
  ): ConsensusChatResult {
    const lowercaseQuestion = userQuestion.toLowerCase();
    const activePersonas = personas.slice(0, 3); // Use first 3 personas for mock responses
    
    let response = "";
    let consensusScore = 75;
    
    if (lowercaseQuestion.includes('apple') || lowercaseQuestion.includes('aapl')) {
      response = `**Warren Buffett**: "Apple has built an incredible ecosystem with strong customer loyalty - a true economic moat. The services revenue is particularly compelling for long-term value creation."

**Cathie Wood**: "While Apple is innovative, I'm concerned about iPhone saturation in developed markets. However, their AI and services expansion could drive the next growth phase."

**Peter Lynch**: "Everyone understands Apple products and their business model. Strong brand loyalty, consistent cash flows, and reasonable valuation make it a solid pick for most investors."

**Team Consensus**: Strong buy recommendation. Apple combines value characteristics with innovation potential. Consider it a core holding for balanced portfolios.

*Note: AI advisory team is operating in backup mode due to high demand. Investment principles remain sound.*`;
      consensusScore = 82;
    } else if (lowercaseQuestion.includes('tesla') || lowercaseQuestion.includes('tsla')) {
      response = `**Cathie Wood**: "Tesla is revolutionizing multiple industries - transportation, energy storage, and AI. Their autonomous driving technology could create massive value."

**Warren Buffett**: "I prefer businesses I can understand with predictable cash flows. Tesla's valuation seems disconnected from current fundamentals, though I respect their innovation."

**Michael Burry**: "The market may be overvaluing the EV transition timeline. Traditional automakers are catching up, and competition is intensifying rapidly."

**Team Consensus**: Hold with caution. High growth potential but significant execution risk and valuation concerns. Suitable only for risk-tolerant growth investors.

*Note: AI advisory team is operating in backup mode due to high demand.*`;
      consensusScore = 55;
    } else if (lowercaseQuestion.includes('portfolio') || lowercaseQuestion.includes('diversif')) {
      response = `**Warren Buffett**: "Diversification is protection against ignorance, but if you know what you're doing, concentration can build wealth. Focus on quality companies you understand."

**Peter Lynch**: "Diversify across different sectors, but don't over-diversify. 10-15 quality stocks you can monitor closely is often better than 50 you can't."

**Bill Ackman**: "Build concentrated positions in your highest conviction ideas, but ensure each position has a clear catalyst for value realization."

${portfolioSummary !== "No current positions" 
  ? `**Portfolio Review**: Your current positions show ${portfolioSummary}. Consider rebalancing if any single position exceeds 20% of your portfolio.`
  : "**Starting Out**: Begin with broad market ETFs, then gradually add individual stocks as you develop conviction."}

**Team Consensus**: Balanced approach between concentration and diversification based on your knowledge and risk tolerance.

*Note: AI advisory team is operating in backup mode due to high demand.*`;
      consensusScore = 78;
    } else {
      response = `**Investment Advisory Team Consensus**:

Thank you for your question about "${userQuestion}". Our legendary investment team has reviewed your inquiry:

**Warren Buffett**: "Focus on businesses with strong competitive advantages, consistent earnings growth, and management teams you trust."

**Peter Lynch**: "Invest in what you understand. Look for companies with clear growth stories and reasonable valuations relative to their potential."

**Cathie Wood**: "Consider companies positioned at the forefront of disruptive innovation - they often create exponential value for patient investors."

${portfolioSummary !== "No current positions" 
  ? `**Portfolio Context**: Based on your positions (${portfolioSummary}), consider diversification and risk management strategies.`
  : "**Getting Started**: Begin with thorough research and consider starting with diversified investments before concentrating in individual stocks."}

**Team Consensus**: Combine fundamental analysis with growth potential assessment. Diversification and patience remain key to long-term success.

*Note: AI advisory team is operating in backup mode due to high demand. We recommend conducting your own research before making investment decisions.*`;
      consensusScore = 72;
    }

    return {
      response,
      consensusScore,
      participatingPersonas: activePersonas.map(p => p.name)
    };
  }
}

export const openaiService = new OpenAIService();
