import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Search, Bell, User, ChartLine } from "lucide-react";

export default function MobileHeader() {
  const [searchQuery, setSearchQuery] = useState("");

  const { data: searchResults = [] } = useQuery({
    queryKey: ['/api/stocks/search', searchQuery],
    enabled: searchQuery.length > 0,
  });

  return (
    <header className="sticky top-0 z-50 glass-effect p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-primary rounded-full flex items-center justify-center">
            <ChartLine className="w-4 h-4 text-primary-foreground" />
          </div>
          <h1 className="text-xl font-semibold">AI Hedge Fund</h1>
        </div>
        <div className="flex items-center space-x-2">
          <Button 
            variant="ghost" 
            size="icon" 
            className="w-10 h-10 rounded-full bg-card"
            data-testid="button-notifications"
          >
            <Bell className="w-4 h-4 text-muted-foreground" />
          </Button>
          <Button 
            variant="ghost" 
            size="icon" 
            className="w-10 h-10 rounded-full bg-card"
            data-testid="button-profile"
          >
            <User className="w-4 h-4 text-muted-foreground" />
          </Button>
        </div>
      </div>
      
      {/* Search Bar */}
      <div className="mt-4 relative">
        <Input
          type="text"
          placeholder="Search stocks (e.g., AAPL, TSLA)"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full bg-card border border-border rounded-lg px-4 py-3 pl-10 text-foreground placeholder-muted-foreground focus:ring-2 focus:ring-primary focus:border-transparent"
          data-testid="input-header-search"
        />
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
        
        {/* Search Results Dropdown */}
        {searchResults.length > 0 && (
          <div className="absolute top-full left-0 right-0 mt-1 bg-card border border-border rounded-lg shadow-lg z-50 max-h-60 overflow-y-auto">
            {searchResults.map((stock: any) => (
              <div
                key={stock.symbol}
                className="p-3 hover:bg-accent cursor-pointer transition-colors border-b border-border last:border-b-0"
                onClick={() => {
                  setSearchQuery("");
                  // Handle stock selection
                }}
                data-testid={`search-result-${stock.symbol}`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-primary/10 rounded-lg flex items-center justify-center">
                      <span className="text-xs font-bold text-primary">{stock.symbol}</span>
                    </div>
                    <div>
                      <div className="font-medium text-sm">{stock.name}</div>
                      <div className="text-xs text-muted-foreground">{stock.symbol}</div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-medium">${stock.price}</div>
                    <div className={`text-xs ${stock.changePercent >= 0 ? 'text-emerald-500' : 'text-destructive'}`}>
                      {stock.changePercent >= 0 ? '+' : ''}{stock.changePercent.toFixed(2)}%
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </header>
  );
}
