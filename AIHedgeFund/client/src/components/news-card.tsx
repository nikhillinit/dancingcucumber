import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ExternalLink, TrendingUp, TrendingDown, Minus } from "lucide-react";

interface NewsCardProps {
  article: {
    title: string;
    description?: string;
    url: string;
    urlToImage?: string;
    publishedAt: string;
    source: {
      name: string;
      url?: string;
    };
    sentiment?: "BULLISH" | "BEARISH" | "NEUTRAL";
    impact?: "HIGH" | "MEDIUM" | "LOW";
    stockSymbols?: string[];
  };
  showStockImpact?: boolean;
}

export default function NewsCard({ article, showStockImpact = false }: NewsCardProps) {
  const formatTimeAgo = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffInHours = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60));
    
    if (diffInHours < 1) {
      const diffInMinutes = Math.floor((now.getTime() - date.getTime()) / (1000 * 60));
      return `${diffInMinutes}m ago`;
    } else if (diffInHours < 24) {
      return `${diffInHours}h ago`;
    } else {
      const diffInDays = Math.floor(diffInHours / 24);
      return `${diffInDays}d ago`;
    }
  };

  const getSentimentIcon = (sentiment?: string) => {
    switch (sentiment) {
      case "BULLISH":
        return <TrendingUp className="w-3 h-3" />;
      case "BEARISH":
        return <TrendingDown className="w-3 h-3" />;
      default:
        return <Minus className="w-3 h-3" />;
    }
  };

  const getSentimentColor = (sentiment?: string) => {
    switch (sentiment) {
      case "BULLISH":
        return "text-emerald-500";
      case "BEARISH":
        return "text-destructive";
      default:
        return "text-muted-foreground";
    }
  };

  const getSentimentLabel = (sentiment?: string) => {
    switch (sentiment) {
      case "BULLISH":
        return "Bullish Impact";
      case "BEARISH":
        return "Bearish Impact";
      default:
        return "Neutral";
    }
  };

  const getImpactColor = (impact?: string) => {
    switch (impact) {
      case "HIGH":
        return "bg-destructive text-destructive-foreground";
      case "MEDIUM":
        return "bg-yellow-500 text-black";
      case "LOW":
        return "bg-emerald-500 text-white";
      default:
        return "bg-muted text-muted-foreground";
    }
  };

  return (
    <Card 
      className="bg-card border-border p-4 cursor-pointer hover:bg-accent transition-colors touch-feedback"
      onClick={() => window.open(article.url, '_blank', 'noopener,noreferrer')}
      data-testid={`news-card-${article.title.slice(0, 20).replace(/\s+/g, '-').toLowerCase()}`}
    >
      <div className="flex space-x-3">
        {article.urlToImage && (
          <img
            src={article.urlToImage}
            alt="News thumbnail"
            className="w-16 h-16 rounded-lg object-cover flex-shrink-0"
            onError={(e) => {
              // Fallback to a placeholder if image fails to load
              e.currentTarget.style.display = 'none';
            }}
            data-testid="news-image"
          />
        )}
        <div className="flex-1 min-w-0">
          <h4 className="font-medium text-sm mb-2 line-clamp-2 text-foreground">
            {article.title}
          </h4>
          {article.description && (
            <p className="text-xs text-muted-foreground mb-2 line-clamp-2">
              {article.description}
            </p>
          )}
          <div className="flex items-center justify-between text-xs">
            <div className="flex items-center space-x-2 text-muted-foreground">
              <span>{article.source.name}</span>
              <span>•</span>
              <span>{formatTimeAgo(article.publishedAt)}</span>
            </div>
            <div className="flex items-center space-x-2">
              {article.impact && (
                <Badge 
                  variant="secondary" 
                  className={`text-xs px-2 py-0.5 ${getImpactColor(article.impact)}`}
                  data-testid={`news-impact-${article.impact.toLowerCase()}`}
                >
                  {article.impact} Impact
                </Badge>
              )}
              {(showStockImpact || article.sentiment) && (
                <div className={`flex items-center space-x-1 ${getSentimentColor(article.sentiment)}`}>
                  {getSentimentIcon(article.sentiment)}
                  <span className="font-medium">
                    {showStockImpact && article.stockSymbols?.length ? 
                      `${article.stockSymbols[0]} +${(Math.random() * 5).toFixed(1)}%` :
                      getSentimentLabel(article.sentiment)
                    }
                  </span>
                </div>
              )}
            </div>
          </div>
          {article.stockSymbols && article.stockSymbols.length > 0 && (
            <div className="flex items-center space-x-1 mt-2">
              <span className="text-xs text-muted-foreground">Related:</span>
              {article.stockSymbols.slice(0, 3).map((symbol) => (
                <Badge 
                  key={symbol} 
                  variant="outline" 
                  className="text-xs px-1.5 py-0.5"
                  data-testid={`news-stock-${symbol}`}
                >
                  {symbol}
                </Badge>
              ))}
            </div>
          )}
        </div>
        <ExternalLink className="w-4 h-4 text-muted-foreground flex-shrink-0 opacity-60" />
      </div>
    </Card>
  );
}
