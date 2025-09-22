import { useQuery } from "@tanstack/react-query";

export default function PersonaAvatars() {
  const { data: personas = [], isLoading } = useQuery({
    queryKey: ['/api/personas'],
  });

  if (isLoading) {
    return (
      <section className="mt-6">
        <div className="h-6 bg-muted rounded w-1/3 mb-4 animate-pulse"></div>
        <div className="flex space-x-3 overflow-x-auto pb-2">
          {[1, 2, 3, 4].map(i => (
            <div key={i} className="flex-shrink-0 animate-pulse">
              <div className="w-16 h-16 rounded-full bg-muted"></div>
              <div className="text-xs text-center mt-2 max-w-16">
                <div className="h-3 bg-muted rounded mb-1"></div>
                <div className="h-3 bg-muted rounded w-8"></div>
              </div>
            </div>
          ))}
        </div>
      </section>
    );
  }

  // Mock scores for display (in real app, these would come from recent analyses)
  const personaScores: Record<string, number> = {
    "Warren Buffett": 92,
    "Cathie Wood": 88,
    "Peter Lynch": 65,
    "Michael Burry": 34,
    "Bill Ackman": 78
  };

  const getPersonaAvatar = (name: string) => {
    const avatars: Record<string, string> = {
      "Warren Buffett": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=100&h=100&fit=crop&crop=face",
      "Cathie Wood": "https://images.unsplash.com/photo-1494790108755-2616b612b786?w=100&h=100&fit=crop&crop=face",
      "Peter Lynch": "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=100&h=100&fit=crop&crop=face",
      "Michael Burry": "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=100&h=100&fit=crop&crop=face",
      "Bill Ackman": "https://images.unsplash.com/photo-1560250097-0b93528c311a?w=100&h=100&fit=crop&crop=face"
    };
    return avatars[name] || "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=100&h=100&fit=crop&crop=face";
  };

  const getScoreColor = (score: number) => {
    if (score >= 80) return "text-emerald-500";
    if (score >= 60) return "text-yellow-500";
    return "text-destructive";
  };

  return (
    <section className="mt-6">
      <h2 className="text-lg font-semibold mb-4">AI Investment Team</h2>
      <div className="flex space-x-3 overflow-x-auto pb-2">
        {personas.map((persona: any) => {
          const score = personaScores[persona.name] || 75;
          const firstName = persona.name.split(' ')[0];
          
          return (
            <div key={persona.id} className="flex-shrink-0">
              <img
                src={getPersonaAvatar(persona.name)}
                alt={`${persona.name} AI`}
                className="w-16 h-16 rounded-full persona-avatar object-cover"
                data-testid={`persona-avatar-${persona.name.replace(' ', '-').toLowerCase()}`}
              />
              <div className="text-xs text-center mt-2 max-w-16">
                <div className="font-medium truncate">{firstName}</div>
                <div className={`${getScoreColor(score)} font-medium`}>
                  {score}%
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
