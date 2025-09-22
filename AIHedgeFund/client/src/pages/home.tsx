import { useQuery } from "@tanstack/react-query";
import MobileHeader from "@/components/mobile-header";
import BottomNavigation from "@/components/bottom-navigation";
import PortfolioSummary from "@/components/portfolio-summary";
import PersonaAvatars from "@/components/persona-avatars";
import StockAnalysisCard from "@/components/stock-analysis-card";
import NewsCard from "@/components/news-card";
import { Button } from "@/components/ui/button";
import { Plus } from "lucide-react";

export default function Home() {
  const { data: analyses = [], isLoading: analysesLoading } = useQuery({
    queryKey: ['/api/analyses/latest'],
  });

  const { data: news = [], isLoading: newsLoading } = useQuery({
    queryKey: ['/api/news'],
  });

  return (
    <div className="min-h-screen bg-background">
      <MobileHeader />
      
      <main className="px-4 pb-20">
        <PortfolioSummary />
        <PersonaAvatars />
        
        {/* Latest Analysis Section */}
        <section className="mt-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold">Latest Analysis</h2>
            <Button variant="ghost" size="sm" className="text-primary" data-testid="link-view-all-analysis">
              View All
            </Button>
          </div>
          
          {analysesLoading ? (
            <div className="space-y-4">
              {[1, 2].map(i => (
                <div key={i} className="bg-card rounded-lg p-4 border border-border animate-pulse">
                  <div className="h-20 bg-muted rounded"></div>
                </div>
              ))}
            </div>
          ) : analyses.length > 0 ? (
            <div className="space-y-4">
              {analyses.slice(0, 3).map((analysis: any) => (
                <StockAnalysisCard 
                  key={analysis.id} 
                  analysis={analysis}
                  stock={analysis.stock}
                  personas={[analysis.persona]}
                />
              ))}
            </div>
          ) : (
            <div className="bg-card rounded-lg p-6 border border-border text-center">
              <p className="text-muted-foreground">No stock analyses yet</p>
              <p className="text-sm text-muted-foreground mt-1">
                Search for a stock to get AI investment opinions
              </p>
            </div>
          )}
        </section>

        {/* Market News Section */}
        <section className="mt-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold">Market News</h2>
            <Button variant="ghost" size="sm" className="text-primary" data-testid="link-see-all-news">
              See All
            </Button>
          </div>
          
          {newsLoading ? (
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
          ) : news.length > 0 ? (
            <div className="space-y-3">
              {news.slice(0, 4).map((article: any, index: number) => (
                <NewsCard key={index} article={article} />
              ))}
            </div>
          ) : (
            <div className="bg-card rounded-lg p-6 border border-border text-center">
              <p className="text-muted-foreground">No news available</p>
            </div>
          )}
        </section>
      </main>

      {/* Floating Action Button */}
      <Button 
        className="fixed bottom-20 right-4 w-14 h-14 rounded-full shadow-lg touch-feedback"
        data-testid="button-add-stock"
      >
        <Plus className="w-6 h-6" />
      </Button>

      <BottomNavigation currentPage="home" />
    </div>
  );
}
