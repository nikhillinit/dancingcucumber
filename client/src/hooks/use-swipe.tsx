import { useEffect, useRef, useState } from "react";

interface SwipeHandlers {
  onSwipeLeft?: () => void;
  onSwipeRight?: () => void;
  onSwipeUp?: () => void;
  onSwipeDown?: () => void;
}

interface SwipeOptions {
  threshold?: number; // Minimum distance for a swipe to be detected
  velocityThreshold?: number; // Minimum velocity for a swipe
  preventDefaultTouchmoveEvent?: boolean;
  delta?: number; // Minimum delta for a swipe
}

const DEFAULT_OPTIONS: Required<SwipeOptions> = {
  threshold: 50,
  velocityThreshold: 0.3,
  preventDefaultTouchmoveEvent: false,
  delta: 10,
};

export function useSwipe(
  handlers: SwipeHandlers,
  options: SwipeOptions = {}
) {
  const { threshold, velocityThreshold, preventDefaultTouchmoveEvent, delta } = {
    ...DEFAULT_OPTIONS,
    ...options,
  };

  const [touchStart, setTouchStart] = useState<{
    x: number;
    y: number;
    time: number;
  } | null>(null);

  const [touchEnd, setTouchEnd] = useState<{
    x: number;
    y: number;
    time: number;
  } | null>(null);

  const elementRef = useRef<HTMLElement>(null);

  const onTouchStart = (e: TouchEvent) => {
    setTouchEnd(null);
    setTouchStart({
      x: e.targetTouches[0].clientX,
      y: e.targetTouches[0].clientY,
      time: Date.now(),
    });
  };

  const onTouchMove = (e: TouchEvent) => {
    if (preventDefaultTouchmoveEvent) {
      e.preventDefault();
    }
  };

  const onTouchEnd = (e: TouchEvent) => {
    if (!touchStart) return;

    setTouchEnd({
      x: e.changedTouches[0].clientX,
      y: e.changedTouches[0].clientY,
      time: Date.now(),
    });
  };

  useEffect(() => {
    if (!touchStart || !touchEnd) return;

    const distanceX = touchStart.x - touchEnd.x;
    const distanceY = touchStart.y - touchEnd.y;
    const timeDelta = touchEnd.time - touchStart.time;
    
    const velocity = Math.sqrt(distanceX * distanceX + distanceY * distanceY) / timeDelta;
    
    const isHorizontalSwipe = Math.abs(distanceX) > Math.abs(distanceY);
    const isValidSwipe = 
      Math.abs(isHorizontalSwipe ? distanceX : distanceY) > threshold &&
      velocity > velocityThreshold;

    if (!isValidSwipe) return;

    if (isHorizontalSwipe) {
      if (distanceX > delta && handlers.onSwipeLeft) {
        handlers.onSwipeLeft();
      } else if (distanceX < -delta && handlers.onSwipeRight) {
        handlers.onSwipeRight();
      }
    } else {
      if (distanceY > delta && handlers.onSwipeUp) {
        handlers.onSwipeUp();
      } else if (distanceY < -delta && handlers.onSwipeDown) {
        handlers.onSwipeDown();
      }
    }
  }, [touchEnd, touchStart, handlers, threshold, velocityThreshold, delta]);

  useEffect(() => {
    const element = elementRef.current;
    if (!element) return;

    element.addEventListener("touchstart", onTouchStart, { passive: true });
    element.addEventListener("touchmove", onTouchMove, { passive: !preventDefaultTouchmoveEvent });
    element.addEventListener("touchend", onTouchEnd, { passive: true });

    return () => {
      element.removeEventListener("touchstart", onTouchStart);
      element.removeEventListener("touchmove", onTouchMove);
      element.removeEventListener("touchend", onTouchEnd);
    };
  }, [preventDefaultTouchmoveEvent]);

  return elementRef;
}

export function useSwipeableDiv(handlers: SwipeHandlers, options?: SwipeOptions) {
  const swipeRef = useSwipe(handlers, options);

  const SwipeableDiv = ({ 
    children, 
    className = "", 
    ...props 
  }: {
    children: React.ReactNode;
    className?: string;
    [key: string]: any;
  }) => (
    <div
      ref={swipeRef}
      className={className}
      {...props}
    >
      {children}
    </div>
  );

  return SwipeableDiv;
}

// Hook for pull-to-refresh functionality
export function usePullToRefresh(onRefresh: () => void | Promise<void>) {
  const [isPulling, setIsPulling] = useState(false);
  const [pullDistance, setPullDistance] = useState(0);
  const elementRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const element = elementRef.current;
    if (!element) return;

    let startY = 0;
    let isPullingDown = false;

    const handleTouchStart = (e: TouchEvent) => {
      if (window.scrollY === 0) {
        startY = e.touches[0].clientY;
        isPullingDown = true;
      }
    };

    const handleTouchMove = (e: TouchEvent) => {
      if (!isPullingDown || window.scrollY > 0) return;

      const currentY = e.touches[0].clientY;
      const distance = currentY - startY;

      if (distance > 0) {
        setIsPulling(true);
        setPullDistance(distance);
        
        // Add some resistance to the pull
        const resistanceDistance = distance * 0.5;
        if (resistanceDistance > 100) {
          e.preventDefault();
        }
      }
    };

    const handleTouchEnd = async () => {
      if (isPulling && pullDistance > 100) {
        try {
          await onRefresh();
        } catch (error) {
          console.error("Refresh failed:", error);
        }
      }
      
      setIsPulling(false);
      setPullDistance(0);
      isPullingDown = false;
    };

    element.addEventListener("touchstart", handleTouchStart, { passive: true });
    element.addEventListener("touchmove", handleTouchMove, { passive: false });
    element.addEventListener("touchend", handleTouchEnd, { passive: true });

    return () => {
      element.removeEventListener("touchstart", handleTouchStart);
      element.removeEventListener("touchmove", handleTouchMove);
      element.removeEventListener("touchend", handleTouchEnd);
    };
  }, [onRefresh, isPulling, pullDistance]);

  return {
    elementRef,
    isPulling,
    pullDistance,
  };
}
