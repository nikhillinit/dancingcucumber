import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import ConsensusBar from "./consensus-bar";

interface StockAnalysisCardProps {
  analysis: any;
  stock: any;
  personas: any[];
  consensusScore?: number;
}

export default function StockAnalysisCard({ 
  analysis, 
  stock, 
  personas,
  consensusScore = 84 
}: StockAnalysisCardProps) {
  const getRecommendationColor = (recommendation: string) => {
    switch (recommendation) {
      case "STRONG_BUY":
      case "BUY":
        return "text-emerald-500";
      case "HOLD":
        return "text-yellow-500";
      case "SELL":
      case "STRONG_SELL":
        return "text-destructive";
      default:
        return "text-muted-foreground";
    }
  };

  const formatRecommendation = (recommendation: string) => {
    return recommendation.replace("_", " ");
  };

  const getPersonaInitials = (name: string) => {
    return name.split(' ').map(n => n[0]).join('').slice(0, 2);
  };

  return (
    <div className="bg-card rounded-lg p-4 border border-border mb-4">
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center space-x-3">
          <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center">
            <span className="text-lg font-bold text-primary">{stock.symbol}</span>
          </div>
          <div>
            <h3 className="font-semibold">{stock.name}</h3>
            <div className="text-sm text-muted-foreground">
              ${parseFloat(stock.currentPrice || '0').toFixed(2)}{' '}
              <span className={parseFloat(stock.priceChangePercent || '0') >= 0 ? 'text-emerald-500' : 'text-destructive'}>
                {parseFloat(stock.priceChangePercent || '0') >= 0 ? '+' : ''}{parseFloat(stock.priceChangePercent || '0').toFixed(1)}%
              </span>
            </div>
          </div>
        </div>
        <div className="text-right">
          <div className="text-sm text-muted-foreground">Consensus</div>
          <div className="text-lg font-semibold text-emerald-500">{consensusScore}%</div>
        </div>
      </div>
      
      {/* Persona Opinions Summary */}
      <div className="space-y-2 mb-4">
        {personas.slice(0, 3).map((persona, index) => (
          <div key={persona.id || index} className="flex items-center justify-between text-sm">
            <div className="flex items-center space-x-2">
              <div className={`w-6 h-6 rounded-full ${
                analysis.recommendation === "BUY" || analysis.recommendation === "STRONG_BUY" 
                  ? "bg-emerald-500/20" 
                  : analysis.recommendation === "HOLD"
                  ? "bg-yellow-500/20"
                  : "bg-destructive/20"
              } flex items-center justify-center`}>
                <span className={`text-xs ${
                  analysis.recommendation === "BUY" || analysis.recommendation === "STRONG_BUY" 
                    ? "text-emerald-500" 
                    : analysis.recommendation === "HOLD"
                    ? "text-yellow-500"
                    : "text-destructive"
                }`}>
                  {getPersonaInitials(persona.name)}
                </span>
              </div>
              <span>{persona.name}</span>
            </div>
            <span className={`font-medium ${getRecommendationColor(analysis.recommendation)}`}>
              {formatRecommendation(analysis.recommendation)}
            </span>
          </div>
        ))}
      </div>
      
      {/* Key Insight */}
      <div className="bg-muted/50 rounded-lg p-3 mb-3">
        <div className="text-sm font-medium mb-1">AI Analysis Summary</div>
        <p className="text-sm text-muted-foreground">
          {analysis.reasoning?.substring(0, 150)}...
        </p>
      </div>
      
      {/* Consensus Bar */}
      <div className="mb-3">
        <ConsensusBar score={consensusScore} />
      </div>
      
      {/* Action Buttons */}
      <div className="flex space-x-2">
        <Button 
          className="flex-1 touch-feedback" 
          size="sm"
          data-testid={`button-view-debate-${stock.symbol}`}
        >
          View Debate
        </Button>
        <Button 
          className="flex-1 bg-emerald-500 hover:bg-emerald-600 text-white touch-feedback" 
          size="sm"
          data-testid={`button-add-portfolio-${stock.symbol}`}
        >
          Add to Portfolio
        </Button>
      </div>
    </div>
  );
}
