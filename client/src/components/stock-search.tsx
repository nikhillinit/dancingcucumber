import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Search, TrendingUp, TrendingDown, Loader2 } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";

interface StockSearchProps {
  onStockSelect?: (stock: any) => void;
  showAnalyzeButton?: boolean;
}

export default function StockSearch({ onStockSelect, showAnalyzeButton = true }: StockSearchProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedStock, setSelectedStock] = useState<any>(null);
  const { toast } = useToast();

  const { data: searchResults = [], isLoading: searchLoading } = useQuery({
    queryKey: ['/api/stocks/search', { q: searchQuery }],
    enabled: searchQuery.length > 0,
  });

  const analyzeStockMutation = useMutation({
    mutationFn: async (symbol: string) => {
      const response = await apiRequest('POST', `/api/stocks/${symbol}/analyze`);
      return response.json();
    },
    onSuccess: (data) => {
      toast({
        title: "Analysis Complete",
        description: `AI analysis for ${data.stock.symbol} has been generated.`,
      });
      queryClient.invalidateQueries({ queryKey: ['/api/analyses/latest'] });
    },
    onError: (error: any) => {
      toast({
        title: "Analysis Failed",
        description: error.message || "Failed to analyze stock",
        variant: "destructive",
      });
    },
  });

  const handleStockSelect = (stock: any) => {
    setSelectedStock(stock);
    setSearchQuery("");
    if (onStockSelect) {
      onStockSelect(stock);
    }
  };

  const handleAnalyze = async (symbol: string) => {
    if (!symbol) return;
    analyzeStockMutation.mutate(symbol);
  };

  return (
    <div className="space-y-4">
      {/* Search Input */}
      <div className="relative">
        <Input
          type="text"
          placeholder="Search stocks (e.g., AAPL, TSLA, NVDA)"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="pl-10 bg-card border-border text-foreground placeholder-muted-foreground focus:ring-2 focus:ring-primary focus:border-transparent"
          data-testid="input-stock-search"
        />
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
      </div>

      {/* Search Results */}
      {searchQuery && (
        <Card className="bg-card border-border">
          {searchLoading ? (
            <div className="p-4 flex items-center justify-center">
              <Loader2 className="w-4 h-4 animate-spin mr-2" />
              <span className="text-muted-foreground">Searching stocks...</span>
            </div>
          ) : searchResults.length > 0 ? (
            <div className="divide-y divide-border">
              {searchResults.map((stock: any) => (
                <div
                  key={stock.symbol}
                  className="p-4 hover:bg-accent cursor-pointer transition-colors touch-feedback"
                  onClick={() => handleStockSelect(stock)}
                  data-testid={`stock-result-${stock.symbol}`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center">
                        <span className="text-sm font-bold text-primary">{stock.symbol}</span>
                      </div>
                      <div>
                        <div className="font-medium text-foreground">{stock.name}</div>
                        <div className="text-sm text-muted-foreground">{stock.symbol}</div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm font-medium text-foreground">
                        ${stock.price?.toFixed(2)}
                      </div>
                      <div className={`text-xs flex items-center ${
                        (stock.changePercent || 0) >= 0 ? 'text-emerald-500' : 'text-destructive'
                      }`}>
                        {(stock.changePercent || 0) >= 0 ? (
                          <TrendingUp className="w-3 h-3 mr-1" />
                        ) : (
                          <TrendingDown className="w-3 h-3 mr-1" />
                        )}
                        {(stock.changePercent || 0) >= 0 ? '+' : ''}{(stock.changePercent || 0).toFixed(2)}%
                      </div>
                    </div>
                  </div>
                  
                  {showAnalyzeButton && (
                    <div className="mt-3 pt-3 border-t border-border">
                      <Button
                        size="sm"
                        className="w-full touch-feedback"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleAnalyze(stock.symbol);
                        }}
                        disabled={analyzeStockMutation.isPending}
                        data-testid={`button-analyze-${stock.symbol}`}
                      >
                        {analyzeStockMutation.isPending ? (
                          <>
                            <Loader2 className="w-4 h-4 animate-spin mr-2" />
                            Analyzing...
                          </>
                        ) : (
                          "Get AI Analysis"
                        )}
                      </Button>
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="p-4 text-center text-muted-foreground">
              No stocks found for "{searchQuery}"
            </div>
          )}
        </Card>
      )}

      {/* Selected Stock */}
      {selectedStock && !searchQuery && (
        <Card className="bg-card border-border p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center">
                <span className="text-lg font-bold text-primary">{selectedStock.symbol}</span>
              </div>
              <div>
                <div className="font-semibold text-foreground">{selectedStock.name}</div>
                <div className="text-sm text-muted-foreground">
                  ${selectedStock.price?.toFixed(2)}{' '}
                  <span className={`${
                    (selectedStock.changePercent || 0) >= 0 ? 'text-emerald-500' : 'text-destructive'
                  }`}>
                    {(selectedStock.changePercent || 0) >= 0 ? '+' : ''}{(selectedStock.changePercent || 0).toFixed(2)}%
                  </span>
                </div>
              </div>
            </div>
            {showAnalyzeButton && (
              <Button
                onClick={() => handleAnalyze(selectedStock.symbol)}
                disabled={analyzeStockMutation.isPending}
                className="touch-feedback"
                data-testid={`button-analyze-selected-${selectedStock.symbol}`}
              >
                {analyzeStockMutation.isPending ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin mr-2" />
                    Analyzing...
                  </>
                ) : (
                  "Analyze"
                )}
              </Button>
            )}
          </div>
        </Card>
      )}
    </div>
  );
}
