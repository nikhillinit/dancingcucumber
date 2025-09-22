import { Link, useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { 
  Home, 
  BarChart3, 
  MessageSquare, 
  Briefcase, 
  Newspaper 
} from "lucide-react";

interface BottomNavigationProps {
  currentPage?: string;
}

export default function BottomNavigation({ currentPage }: BottomNavigationProps) {
  const [location] = useLocation();
  
  const isActive = (path: string) => {
    if (path === "/" && location === "/") return true;
    if (path !== "/" && location.startsWith(path)) return true;
    return false;
  };

  const navItems = [
    { path: "/", icon: Home, label: "Home", testId: "nav-home" },
    { path: "/analysis", icon: BarChart3, label: "Analysis", testId: "nav-analysis" },
    { path: "/debates", icon: MessageSquare, label: "Debates", testId: "nav-debates" },
    { path: "/portfolio", icon: Briefcase, label: "Portfolio", testId: "nav-portfolio" },
    { path: "/news", icon: Newspaper, label: "News", testId: "nav-news" },
  ];

  return (
    <nav className="fixed bottom-0 left-0 right-0 bg-card border-t border-border">
      <div className="flex items-center justify-around py-2">
        {navItems.map(({ path, icon: Icon, label, testId }) => (
          <Link key={path} href={path}>
            <Button
              variant="ghost"
              size="sm"
              className={`flex flex-col items-center py-2 px-3 h-auto space-y-1 ${
                isActive(path) ? "text-primary" : "text-muted-foreground"
              }`}
              data-testid={testId}
            >
              <Icon className="w-5 h-5" />
              <span className="text-xs">{label}</span>
            </Button>
          </Link>
        ))}
      </div>
    </nav>
  );
}
