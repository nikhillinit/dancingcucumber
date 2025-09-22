import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import MobileHeader from "@/components/mobile-header";
import BottomNavigation from "@/components/bottom-navigation";
import NewsCard from "@/components/news-card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default function News() {
  const [activeTab, setActiveTab] = useState("market");

  const { data: marketNews = [], isLoading: marketLoading } = useQuery({
    queryKey: ['/api/news'],
  });

  const { data: stockNews = [], isLoading: stockLoading } = useQuery({
    queryKey: ['/api/news/stocks/AAPL'], // Example stock news
    enabled: activeTab === "stocks",
  });

  return (
    <div className="min-h-screen bg-background">
      <MobileHeader />
      
      <main className="px-4 pb-20">
        <section className="mt-6">
          <h2 className="text-lg font-semibold mb-4">Market News</h2>
          
          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="market" data-testid="tab-market-news">Market</TabsTrigger>
              <TabsTrigger value="stocks" data-testid="tab-stock-news">Stocks</TabsTrigger>
              <TabsTrigger value="analysis" data-testid="tab-analysis-news">Analysis</TabsTrigger>
            </TabsList>
            
            <TabsContent value="market" className="mt-6">
              {marketLoading ? (
                <div className="space-y-3">
                  {[1, 2, 3, 4, 5].map(i => (
                    <div key={i} className="bg-card rounded-lg p-4 border border-border animate-pulse">
                      <div className="flex space-x-3">
                        <div className="w-16 h-16 bg-muted rounded-lg"></div>
                        <div className="flex-1 space-y-2">
                          <div className="h-4 bg-muted rounded w-3/4"></div>
                          <div className="h-3 bg-muted rounded w-full"></div>
                          <div className="h-3 bg-muted rounded w-1/2"></div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : marketNews.length > 0 ? (
                <div className="space-y-3">
                  {marketNews.map((article: any, index: number) => (
                    <NewsCard key={index} article={article} />
                  ))}
                </div>
              ) : (
                <div className="bg-card rounded-lg p-8 border border-border text-center">
                  <p className="text-muted-foreground">No market news available</p>
                </div>
              )}
            </TabsContent>
            
            <TabsContent value="stocks" className="mt-6">
              {stockLoading ? (
                <div className="space-y-3">
                  {[1, 2, 3].map(i => (
                    <div key={i} className="bg-card rounded-lg p-4 border border-border animate-pulse">
                      <div className="flex space-x-3">
                        <div className="w-16 h-16 bg-muted rounded-lg"></div>
                        <div className="flex-1 space-y-2">
                          <div className="h-4 bg-muted rounded w-3/4"></div>
                          <div className="h-3 bg-muted rounded w-full"></div>
                          <div className="h-3 bg-muted rounded w-1/2"></div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : stockNews.length > 0 ? (
                <div className="space-y-3">
                  {stockNews.map((article: any, index: number) => (
                    <NewsCard key={index} article={article} showStockImpact />
                  ))}
                </div>
              ) : (
                <div className="bg-card rounded-lg p-8 border border-border text-center">
                  <p className="text-muted-foreground">No stock-specific news available</p>
                </div>
              )}
            </TabsContent>
            
            <TabsContent value="analysis" className="mt-6">
              <div className="bg-card rounded-lg p-8 border border-border text-center">
                <p className="text-muted-foreground">AI-powered news analysis coming soon</p>
              </div>
            </TabsContent>
          </Tabs>
        </section>

        {/* News Categories */}
        <section className="mt-8">
          <h2 className="text-lg font-semibold mb-4">Categories</h2>
          <div className="grid grid-cols-2 gap-3">
            <Button variant="outline" className="justify-start h-auto p-4">
              <div className="text-left">
                <div className="font-medium">Technology</div>
                <div className="text-sm text-muted-foreground">12 articles</div>
              </div>
            </Button>
            <Button variant="outline" className="justify-start h-auto p-4">
              <div className="text-left">
                <div className="font-medium">Finance</div>
                <div className="text-sm text-muted-foreground">8 articles</div>
              </div>
            </Button>
            <Button variant="outline" className="justify-start h-auto p-4">
              <div className="text-left">
                <div className="font-medium">Energy</div>
                <div className="text-sm text-muted-foreground">5 articles</div>
              </div>
            </Button>
            <Button variant="outline" className="justify-start h-auto p-4">
              <div className="text-left">
                <div className="font-medium">Healthcare</div>
                <div className="text-sm text-muted-foreground">3 articles</div>
              </div>
            </Button>
          </div>
        </section>
      </main>

      <BottomNavigation currentPage="news" />
    </div>
  );
}
