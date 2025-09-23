import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import MobileHeader from "@/components/mobile-header";
import BottomNavigation from "@/components/bottom-navigation";
import ConsensusBar from "@/components/consensus-bar";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { TrendingUp, TrendingDown, Plus } from "lucide-react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";

const addPositionSchema = z.object({
  symbol: z.string().min(1, "Stock symbol is required").max(10, "Symbol too long"),
  shares: z.string().transform(val => parseFloat(val)).refine(val => val > 0, "Shares must be greater than 0"),
  price: z.string().transform(val => parseFloat(val)).refine(val => val > 0, "Price must be greater than 0")
});

type AddPositionFormData = z.infer<typeof addPositionSchema>;

export default function Portfolio() {
  const [showAddPosition, setShowAddPosition] = useState(false);
  const { toast } = useToast();
  const { data: portfolio, isLoading } = useQuery({
    queryKey: ['/api/portfolio'],
  });

  const positions = portfolio?.positions || [];
  const summary = portfolio?.summary || {
    totalValue: 0,
    totalReturn: 0,
    totalReturnPercent: 0,
    consensusScore: 0
  };

  const form = useForm<AddPositionFormData>({
    resolver: zodResolver(addPositionSchema),
    defaultValues: {
      symbol: "",
      shares: "",
      price: ""
    }
  });

  const addPositionMutation = useMutation({
    mutationFn: async (data: AddPositionFormData) => {
      const response = await apiRequest('POST', '/api/portfolio', data);
      return response.json();
    },
    onSuccess: () => {
      toast({
        title: "Position Added",
        description: "Stock position has been added to your portfolio.",
      });
      setShowAddPosition(false);
      form.reset();
      queryClient.invalidateQueries({ queryKey: ['/api/portfolio'] });
    },
    onError: (error: any) => {
      toast({
        title: "Failed to Add Position",
        description: error.message || "Failed to add position to portfolio",
        variant: "destructive",
      });
    },
  });

  const handleAddPosition = () => {
    setShowAddPosition(true);
  };

  const onSubmit = (data: AddPositionFormData) => {
    addPositionMutation.mutate(data);
  };

  return (
    <div className="min-h-screen bg-background">
      <MobileHeader />
      
      <main className="px-4 pb-20">
        {/* Portfolio Summary */}
        <section className="mt-6">
          <div className="bg-card rounded-lg p-4 border border-border">
            <h2 className="text-lg font-semibold mb-3">Portfolio Performance</h2>
            <div className="flex items-center justify-between mb-4">
              <div>
                <div className={`text-2xl font-bold ${summary.totalReturn >= 0 ? 'text-emerald-500' : 'text-destructive'}`}>
                  ${summary.totalValue.toLocaleString()}
                </div>
                <div className={`text-sm flex items-center ${summary.totalReturn >= 0 ? 'text-emerald-500' : 'text-destructive'}`}>
                  {summary.totalReturn >= 0 ? (
                    <TrendingUp className="w-4 h-4 mr-1" />
                  ) : (
                    <TrendingDown className="w-4 h-4 mr-1" />
                  )}
                  <span>
                    {summary.totalReturnPercent >= 0 ? '+' : ''}{summary.totalReturnPercent.toFixed(2)}% 
                    ({summary.totalReturn >= 0 ? '+' : ''}${summary.totalReturn.toLocaleString()})
                  </span>
                </div>
              </div>
              <div className="text-right">
                <div className="text-sm text-muted-foreground">AI Consensus</div>
                <div className="text-lg font-semibold text-emerald-500">
                  {summary.consensusScore.toFixed(0)}%
                </div>
              </div>
            </div>
            
            <ConsensusBar score={summary.consensusScore} />
          </div>
        </section>

        {/* Holdings */}
        <section className="mt-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold">Holdings</h2>
            <Button size="sm" onClick={handleAddPosition} data-testid="button-add-position">
              <Plus className="w-4 h-4 mr-2" />
              Add Position
            </Button>
          </div>
          
          {isLoading ? (
            <div className="space-y-3">
              {[1, 2, 3].map(i => (
                <div key={i} className="bg-card rounded-lg p-4 border border-border animate-pulse">
                  <div className="h-16 bg-muted rounded"></div>
                </div>
              ))}
            </div>
          ) : positions.length > 0 ? (
            <div className="space-y-3">
              {positions.map((position: any) => (
                <div key={position.id} className="bg-card rounded-lg p-4 border border-border">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center space-x-3">
                      <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center">
                        <span className="text-lg font-bold text-primary">
                          {position.stock?.symbol}
                        </span>
                      </div>
                      <div>
                        <h3 className="font-semibold">{position.stock?.name}</h3>
                        <div className="text-sm text-muted-foreground">
                          {parseFloat(position.shares).toFixed(2)} shares @ ${parseFloat(position.averagePrice).toFixed(2)}
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className={`font-semibold ${(position.returnPercent || 0) >= 0 ? 'text-emerald-500' : 'text-destructive'}`}>
                        {(position.returnPercent || 0) >= 0 ? '+' : ''}{parseFloat(position.returnPercent || '0').toFixed(2)}%
                      </div>
                      <div className="text-sm text-muted-foreground">
                        ${parseFloat(position.currentValue || '0').toLocaleString()}
                      </div>
                    </div>
                  </div>
                  
                  {position.consensusScore && (
                    <div className="mt-3">
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-muted-foreground">AI Consensus</span>
                        <span className="font-medium">{position.consensusScore}%</span>
                      </div>
                      <ConsensusBar score={position.consensusScore} size="sm" />
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="bg-card rounded-lg p-8 border border-border text-center">
              <div className="text-muted-foreground mb-4">Your portfolio is empty</div>
              <Button onClick={handleAddPosition} data-testid="button-start-investing">
                <Plus className="w-4 h-4 mr-2" />
                Start Investing
              </Button>
            </div>
          )}
        </section>

        {/* Performance Metrics */}
        {positions.length > 0 && (
          <section className="mt-6">
            <h2 className="text-lg font-semibold mb-4">Performance Metrics</h2>
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-card rounded-lg p-4 border border-border text-center">
                <div className="text-2xl font-bold text-emerald-500">
                  {positions.length}
                </div>
                <div className="text-sm text-muted-foreground">Positions</div>
              </div>
              <div className="bg-card rounded-lg p-4 border border-border text-center">
                <div className="text-2xl font-bold text-primary">
                  {summary.consensusScore.toFixed(0)}%
                </div>
                <div className="text-sm text-muted-foreground">Avg Consensus</div>
              </div>
            </div>
          </section>
        )}
      </main>

      <BottomNavigation currentPage="portfolio" />
      
      {/* Add Position Dialog */}
      <Dialog open={showAddPosition} onOpenChange={setShowAddPosition}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Add Position</DialogTitle>
          </DialogHeader>
          <Form {...form}>
            <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
              <FormField
                control={form.control}
                name="symbol"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Stock Symbol</FormLabel>
                    <FormControl>
                      <Input placeholder="AAPL" {...field} data-testid="input-symbol" />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="shares"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Number of Shares</FormLabel>
                    <FormControl>
                      <Input type="number" step="0.01" placeholder="100" {...field} data-testid="input-shares" />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="price"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Purchase Price</FormLabel>
                    <FormControl>
                      <Input type="number" step="0.01" placeholder="150.00" {...field} data-testid="input-price" />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <div className="flex justify-end space-x-2">
                <Button type="button" variant="outline" onClick={() => setShowAddPosition(false)} data-testid="button-cancel">
                  Cancel
                </Button>
                <Button type="submit" disabled={addPositionMutation.isPending} data-testid="button-save-position">
                  {addPositionMutation.isPending ? "Adding..." : "Add Position"}
                </Button>
              </div>
            </form>
          </Form>
        </DialogContent>
      </Dialog>
    </div>
  );
}
