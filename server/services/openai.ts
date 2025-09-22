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
      // Return a fallback response instead of throwing an error
      return {
        response: "I apologize, but I'm having trouble connecting with the investment team right now. Please try your question again.",
        consensusScore: 0,
        participatingPersonas: []
      };
    }
  }
}

export const openaiService = new OpenAIService();
