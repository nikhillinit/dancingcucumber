import { useQuery } from "@tanstack/react-query";
import MobileHeader from "@/components/mobile-header";
import BottomNavigation from "@/components/bottom-navigation";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { MessageSquare, Users, TrendingUp } from "lucide-react";

export default function Debates() {
  const { data: debates = [], isLoading } = useQuery({
    queryKey: ['/api/debates'],
  });

  return (
    <div className="min-h-screen bg-background">
      <MobileHeader />
      
      <main className="px-4 pb-20">
        <section className="mt-6">
          <h2 className="text-lg font-semibold mb-4">Active Debates</h2>
          
          {isLoading ? (
            <div className="space-y-4">
              {[1, 2, 3].map(i => (
                <div key={i} className="bg-card rounded-lg p-4 border border-border animate-pulse">
                  <div className="h-24 bg-muted rounded"></div>
                </div>
              ))}
            </div>
          ) : debates.length > 0 ? (
            <div className="space-y-4">
              {debates.map((debate: any) => (
                <div key={debate.id} className="bg-card rounded-lg p-4 border border-border">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex-1">
                      <h3 className="font-semibold mb-2">{debate.title}</h3>
                      <div className="flex items-center space-x-4 text-sm text-muted-foreground">
                        <div className="flex items-center space-x-1">
                          <Users className="w-4 h-4" />
                          <span>{debate.participants?.length || 0} investors</span>
                        </div>
                        <div className="flex items-center space-x-1">
                          <MessageSquare className="w-4 h-4" />
                          <span>{debate.messages?.length || 0} messages</span>
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <Badge variant={debate.status === "ACTIVE" ? "default" : "secondary"}>
                        {debate.status}
                      </Badge>
                      {debate.consensusScore && (
                        <div className="text-sm text-muted-foreground mt-1">
                          Consensus: {debate.consensusScore}%
                        </div>
                      )}
                    </div>
                  </div>
                  
                  {debate.messages && debate.messages.length > 0 && (
                    <div className="bg-muted/50 rounded-lg p-3 mb-3">
                      <div className="text-sm font-medium mb-1">Latest Message</div>
                      <p className="text-sm text-muted-foreground">
                        {debate.messages[debate.messages.length - 1]?.message}
                      </p>
                    </div>
                  )}
                  
                  <div className="flex space-x-2">
                    <Button 
                      className="flex-1" 
                      size="sm"
                      data-testid={`button-join-debate-${debate.id}`}
                    >
                      <MessageSquare className="w-4 h-4 mr-2" />
                      Join Debate
                    </Button>
                    <Button 
                      variant="outline" 
                      size="sm"
                      data-testid={`button-view-debate-${debate.id}`}
                    >
                      View Details
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="bg-card rounded-lg p-8 border border-border text-center">
              <MessageSquare className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
              <div className="text-muted-foreground mb-2">No active debates</div>
              <p className="text-sm text-muted-foreground">
                Debates will appear when AI investors have different opinions on stocks
              </p>
            </div>
          )}
        </section>

        {/* Featured Debate Topics */}
        <section className="mt-8">
          <h2 className="text-lg font-semibold mb-4">Trending Topics</h2>
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-card rounded-lg p-4 border border-border text-center">
              <TrendingUp className="w-8 h-8 text-emerald-500 mx-auto mb-2" />
              <div className="text-sm font-medium">AI Stocks</div>
              <div className="text-xs text-muted-foreground">3 debates</div>
            </div>
            <div className="bg-card rounded-lg p-4 border border-border text-center">
              <TrendingUp className="w-8 h-8 text-yellow-500 mx-auto mb-2" />
              <div className="text-sm font-medium">EV Market</div>
              <div className="text-xs text-muted-foreground">2 debates</div>
            </div>
            <div className="bg-card rounded-lg p-4 border border-border text-center">
              <TrendingUp className="w-8 h-8 text-blue-500 mx-auto mb-2" />
              <div className="text-sm font-medium">Banking</div>
              <div className="text-xs text-muted-foreground">1 debate</div>
            </div>
            <div className="bg-card rounded-lg p-4 border border-border text-center">
              <TrendingUp className="w-8 h-8 text-purple-500 mx-auto mb-2" />
              <div className="text-sm font-medium">Energy</div>
              <div className="text-xs text-muted-foreground">1 debate</div>
            </div>
          </div>
        </section>
      </main>

      <BottomNavigation currentPage="debates" />
    </div>
  );
}
