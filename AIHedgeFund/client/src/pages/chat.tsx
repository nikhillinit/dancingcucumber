import { useState, useEffect, useRef } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import MobileHeader from "@/components/mobile-header";
import BottomNavigation from "@/components/bottom-navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Send, MessageSquare, Users, TrendingUp } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import type { ChatConversation, ChatMessage } from "@shared/schema";

export default function Chat() {
  const [message, setMessage] = useState("");
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();

  // Get or create a conversation
  const { data: conversations = [] } = useQuery<ChatConversation[]>({
    queryKey: ['/api/chat/conversations'],
  });

  // Messages query with proper URL - always provide stable key, use enabled to control fetching
  const messagesKey = `/api/chat/conversations/${currentConversationId ?? 'pending'}/messages`;
  const { data: messages = [] } = useQuery<ChatMessage[]>({
    queryKey: [messagesKey],
    enabled: !!currentConversationId,
  });

  const createConversationMutation = useMutation({
    mutationFn: async (data: { title: string; status: string }) => {
      const response = await apiRequest('POST', '/api/chat/conversations', data);
      return response.json();
    },
    onSuccess: (data) => {
      setCurrentConversationId(data.id);
      queryClient.invalidateQueries({ queryKey: ['/api/chat/conversations'] });
    },
  });

  // Initialize conversation if none exists
  useEffect(() => {
    if (conversations.length === 0 && !createConversationMutation.isPending) {
      createConversationMutation.mutate({
        title: "Investment Advisory Chat",
        status: "ACTIVE"
      });
    } else if (conversations.length > 0 && !currentConversationId) {
      setCurrentConversationId(conversations[0].id);
    }
  }, [conversations.length, currentConversationId, createConversationMutation.isPending]);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessageMutation = useMutation({
    mutationFn: async (data: { role: string; content: string }) => {
      const response = await apiRequest('POST', `/api/chat/conversations/${currentConversationId}/messages`, data);
      return response.json();
    },
    onMutate: async (newMessage: { role: string; content: string }) => {
      const queryKey = [messagesKey];
      
      // Cancel any outgoing refetches (so they don't overwrite our optimistic update)
      await queryClient.cancelQueries({ queryKey });

      // Snapshot the previous value
      const previousMessages = queryClient.getQueryData<ChatMessage[]>(queryKey);

      // Optimistically update to the new value
      const optimisticMessage: ChatMessage = {
        id: `temp-${Date.now()}`,
        conversationId: currentConversationId!,
        role: newMessage.role,
        content: newMessage.content,
        metadata: null,
        createdAt: new Date()
      };

      queryClient.setQueryData<ChatMessage[]>(queryKey, old => [...(old || []), optimisticMessage]);

      // Return a context object with the snapshotted value
      return { previousMessages, queryKey };
    },
    onSuccess: (data) => {
      // Replace the optimistic update with the real data
      queryClient.invalidateQueries({ queryKey: [messagesKey] });
      
      if (data.aiMessage) {
        toast({
          title: "Advisory Team Response",
          description: "The investment team has provided their consensus opinion.",
        });
      } else if (data.userMessage) {
        // AI service failed but we got the user message - check if we should show a fallback
        toast({
          title: "AI Service Unavailable",
          description: "Your message was sent, but the AI advisory team is temporarily unavailable.",
          variant: "default",
        });
      }
    },
    onError: (error: any, newMessage, context) => {
      // If the mutation fails, use the context returned from onMutate to roll back
      if (context?.queryKey) {
        queryClient.setQueryData(context.queryKey, context.previousMessages);
      }
      
      toast({
        title: "Message Failed",
        description: error.message || "Failed to send message",
        variant: "destructive",
      });
    },
  });

  const handleSendMessage = async () => {
    if (!message.trim() || !currentConversationId) return;

    const messageText = message.trim();
    setMessage("");

    sendMessageMutation.mutate({
      role: "user",
      content: messageText,
    });
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const formatMessage = (content: string | null | undefined) => {
    if (!content) {
      console.warn("Message content is empty:", content);
      return "No content";
    }
    return content.split('\n').map((line, index) => (
      <span key={index}>
        {line}
        {index < content.split('\n').length - 1 && <br />}
      </span>
    ));
  };

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <MobileHeader />
      
      <main className="flex-1 flex flex-col px-4">
        {/* Chat Header */}
        <div className="mt-6 mb-4">
          <div className="flex items-center space-x-3 mb-2">
            <div className="w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center">
              <Users className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h2 className="text-lg font-semibold">AI Investment Advisory Team</h2>
              <p className="text-sm text-muted-foreground">
                Ask questions and get consensus advice from legendary investors
              </p>
            </div>
          </div>
          
          <div className="flex flex-wrap gap-2 mt-3">
            <Badge variant="outline" className="text-xs">
              <TrendingUp className="w-3 h-3 mr-1" />
              Portfolio Analysis
            </Badge>
            <Badge variant="outline" className="text-xs">Warren Buffett</Badge>
            <Badge variant="outline" className="text-xs">Cathie Wood</Badge>
            <Badge variant="outline" className="text-xs">Peter Lynch</Badge>
            <Badge variant="outline" className="text-xs">+2 more</Badge>
          </div>
        </div>

        {/* Messages Area */}
        <Card className="flex-1 mb-4">
          <CardContent className="p-0 h-full">
            <ScrollArea className="h-96 p-4">
              {messages.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full text-center space-y-4">
                  <MessageSquare className="w-12 h-12 text-muted-foreground" />
                  <div>
                    <h3 className="font-medium mb-2">Welcome to the AI Investment Advisory</h3>
                    <p className="text-sm text-muted-foreground">
                      Ask questions about your portfolio, market trends, or get investment advice from our AI investment team.
                    </p>
                  </div>
                  <div className="bg-muted rounded-lg p-3 text-sm">
                    <p className="font-medium mb-1">Try asking:</p>
                    <ul className="text-muted-foreground space-y-1">
                      <li>• "What do you think about my current portfolio?"</li>
                      <li>• "Should I buy more tech stocks?"</li>
                      <li>• "What are the risks in the current market?"</li>
                    </ul>
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  {messages.map((msg: any, index: number) => {
                    console.log("Rendering message:", msg);
                    return (
                      <div
                        key={msg.id || `msg-${index}`}
                        className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                      >
                        <div
                          className={`max-w-[85%] rounded-lg p-3 ${
                            msg.role === 'user'
                              ? 'bg-primary text-primary-foreground ml-12'
                              : 'bg-muted mr-12'
                          }`}
                        >
                          {msg.role === 'assistant' && (
                            <div className="flex items-center space-x-2 mb-2">
                              <div className="w-6 h-6 bg-primary/10 rounded-full flex items-center justify-center">
                                <Users className="w-3 h-3 text-primary" />
                              </div>
                              <span className="text-xs font-medium">Investment Advisory Team</span>
                              {msg.metadata?.consensusScore && (
                                <Badge variant="secondary" className="text-xs">
                                  {msg.metadata.consensusScore}% Consensus
                                </Badge>
                              )}
                            </div>
                          )}
                          <div className="text-sm">
                            {formatMessage(msg.content)}
                          </div>
                          <div className="text-xs opacity-70 mt-2">
                            {new Date(msg.createdAt).toLocaleTimeString([], { 
                              hour: '2-digit', 
                              minute: '2-digit' 
                            })}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                  <div ref={messagesEndRef} />
                </div>
              )}
            </ScrollArea>
          </CardContent>
        </Card>

        {/* Message Input */}
        <div className="mb-20">
          <div className="flex space-x-2">
            <Input
              placeholder="Ask the investment team a question..."
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={sendMessageMutation.isPending}
              className="flex-1"
              data-testid="input-chat-message"
            />
            <Button 
              onClick={handleSendMessage}
              disabled={!message.trim() || sendMessageMutation.isPending || !currentConversationId}
              data-testid="button-send-message"
            >
              {sendMessageMutation.isPending ? (
                <div className="w-4 h-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
              ) : (
                <Send className="w-4 h-4" />
              )}
            </Button>
          </div>
          
          {sendMessageMutation.isPending && (
            <div className="mt-2 text-sm text-muted-foreground flex items-center space-x-2">
              <div className="w-3 h-3 animate-spin rounded-full border border-primary border-t-transparent" />
              <span>The investment team is analyzing your question...</span>
            </div>
          )}
        </div>
      </main>

      <BottomNavigation currentPage="chat" />
    </div>
  );
}