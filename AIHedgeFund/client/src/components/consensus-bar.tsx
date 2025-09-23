interface ConsensusBarProps {
  score: number; // 0-100
  size?: "sm" | "md";
}

export default function ConsensusBar({ score, size = "md" }: ConsensusBarProps) {
  const height = size === "sm" ? "h-1.5" : "h-2";
  
  return (
    <div className="space-y-2">
      <div className="flex justify-between text-sm text-muted-foreground">
        <span>Bearish</span>
        <span>Neutral</span>
        <span>Bullish</span>
      </div>
      <div className={`${height} bg-muted rounded-full overflow-hidden`}>
        <div 
          className="consensus-bar h-full rounded-full transition-all duration-300 ease-in-out" 
          style={{ width: `${score}%` }}
        ></div>
      </div>
      {size === "md" && (
        <div className="text-center text-sm text-muted-foreground">
          {score}% Consensus Score
        </div>
      )}
    </div>
  );
}
