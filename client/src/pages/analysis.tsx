import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import MobileHeader from "@/components/mobile-header";
import BottomNavigation from "@/components/bottom-navigation";
import StockAnalysisCard from "@/components/stock-analysis-card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Search, Loader2 } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";

export default function Analysis() {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedStock, setSelectedStock] = useState<string | null>(null);
  const { toast } = useToast();

  const { data: analyses = [], isLoading } = useQuery({
    queryKey: ['/api/analyses/latest'],
  });

  const { data: searchResults = [] } = useQuery({
    queryKey: ['/api/stocks/search', searchQuery],
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

  const handleAnalyze = async (symbol: string) => {
    if (!symbol) return;
    analyzeStockMutation.mutate(symbol);
  };

  return (
    <div className="min-h-screen bg-background">
      <MobileHeader />
      
      <main className="px-4 pb-20">
        {/* Search Section */}
        <section className="mt-6">
          <div className="relative">
            <Input
              type="text"
              placeholder="Search stocks (e.g., AAPL, TSLA)"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
              data-testid="input-stock-search"
            />
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          </div>

          {searchResults.length > 0 && (
            <div className="mt-4 space-y-2">
              {searchResults.map((stock: any) => (
                <div
                  key={stock.symbol}
                  className="bg-card rounded-lg p-3 border border-border cursor-pointer hover:bg-accent transition-colors"
                  onClick={() => setSelectedStock(stock.symbol)}
                  data-testid={`stock-result-${stock.symbol}`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center">
                        <span className="text-sm font-bold text-primary">{stock.symbol}</span>
                      </div>
                      <div>
                        <div className="font-medium">{stock.name}</div>
                        <div className="text-sm text-muted-foreground">
                          ${stock.price} {stock.changePercent > 0 ? '+' : ''}{stock.changePercent.toFixed(2)}%
                        </div>
                      </div>
                    </div>
                    <Button 
                      size="sm" 
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
                        "Analyze"
                      )}
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>

        {/* Analysis Results */}
        <section className="mt-8">
          <h2 className="text-lg font-semibold mb-4">Recent Analyses</h2>
          
          {isLoading ? (
            <div className="space-y-4">
              {[1, 2, 3].map(i => (
                <div key={i} className="bg-card rounded-lg p-4 border border-border animate-pulse">
                  <div className="h-32 bg-muted rounded"></div>
                </div>
              ))}
            </div>
          ) : analyses.length > 0 ? (
            <div className="space-y-4">
              {analyses.map((analysis: any) => (
                <StockAnalysisCard 
                  key={analysis.id} 
                  analysis={analysis}
                  stock={analysis.stock}
                  personas={[analysis.persona]}
                />
              ))}
            </div>
          ) : (
            <div className="bg-card rounded-lg p-8 border border-border text-center">
              <div className="text-muted-foreground mb-2">No analyses yet</div>
              <p className="text-sm text-muted-foreground">
                Search for a stock above to get started with AI analysis
              </p>
            </div>
          )}
        </section>
      </main>

      <BottomNavigation currentPage="analysis" />
    </div>
  );
}
