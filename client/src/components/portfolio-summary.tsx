import { useQuery } from "@tanstack/react-query";
import ConsensusBar from "./consensus-bar";
import { TrendingUp, TrendingDown } from "lucide-react";

export default function PortfolioSummary() {
  const { data: portfolio, isLoading } = useQuery({
    queryKey: ['/api/portfolio'],
  });

  if (isLoading) {
    return (
      <section className="mt-6">
        <div className="bg-card rounded-lg p-4 border border-border animate-pulse">
          <div className="h-6 bg-muted rounded w-1/3 mb-3"></div>
          <div className="h-8 bg-muted rounded w-1/2 mb-2"></div>
          <div className="h-4 bg-muted rounded w-1/4 mb-4"></div>
          <div className="h-4 bg-muted rounded w-full"></div>
        </div>
      </section>
    );
  }

  const summary = portfolio?.summary || {
    totalValue: 127543.20,
    totalReturn: 3024.15,
    totalReturnPercent: 2.4,
    consensusScore: 87
  };

  const isPositive = summary.totalReturn >= 0;

  return (
    <section className="mt-6">
      <div className="bg-card rounded-lg p-4 border border-border">
        <h2 className="text-lg font-semibold mb-3">Portfolio Performance</h2>
        <div className="flex items-center justify-between mb-4">
          <div>
            <div className={`text-2xl font-bold ${isPositive ? 'text-emerald-500' : 'text-destructive'}`}>
              ${summary.totalValue.toLocaleString()}
            </div>
            <div className={`text-sm flex items-center ${isPositive ? 'text-emerald-500' : 'text-destructive'}`}>
              {isPositive ? (
                <TrendingUp className="w-4 h-4 mr-1" />
              ) : (
                <TrendingDown className="w-4 h-4 mr-1" />
              )}
              <span>
                {isPositive ? '+' : ''}{summary.totalReturnPercent.toFixed(1)}% 
                ({isPositive ? '+' : ''}${summary.totalReturn.toLocaleString()})
              </span>
            </div>
          </div>
          <div className="text-right">
            <div className="text-sm text-muted-foreground">AI Consensus</div>
            <div className="text-lg font-semibold text-emerald-500">
              {summary.consensusScore}%
            </div>
          </div>
        </div>
        
        <ConsensusBar score={summary.consensusScore} />
      </div>
    </section>
  );
}
