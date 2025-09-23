# Overview

This is an AI-powered hedge fund simulation application that creates a realistic investment advisory platform where AI investor personas analyze stocks and engage in debates. The system features multiple legendary investor personalities (Warren Buffett, Cathie Wood, Peter Lynch, etc.) that provide personalized stock analysis, participate in investment debates, and offer portfolio recommendations. The application combines real-time financial data with AI-generated insights to create an engaging investment education and analysis tool.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Framework**: React with TypeScript using Vite as the build tool
- **Routing**: Wouter for lightweight client-side routing
- **State Management**: TanStack Query (React Query) for server state management and caching
- **UI Framework**: Shadcn/ui components built on Radix UI primitives
- **Styling**: Tailwind CSS with custom design tokens and dark theme support
- **Mobile-First Design**: Responsive layout optimized for mobile with bottom navigation

## Backend Architecture
- **Runtime**: Node.js with Express.js web framework
- **Language**: TypeScript with ES modules
- **API Design**: RESTful API endpoints for stocks, personas, analyses, debates, portfolio, and news
- **Error Handling**: Centralized error middleware with proper HTTP status codes
- **Development**: Hot module replacement via Vite integration for development mode

## Data Storage
- **Database**: PostgreSQL with Drizzle ORM for type-safe database operations
- **Connection**: Neon Database (serverless PostgreSQL) via `@neondatabase/serverless`
- **Schema Management**: Drizzle Kit for migrations and schema management
- **Data Modeling**: Structured tables for investor personas, stocks, analyses, debates, portfolio positions, and news articles

## AI Integration
- **Provider**: OpenAI API for generating persona-based stock analyses and debate content
- **Analysis Engine**: Each investor persona provides unique investment recommendations with confidence scores
- **Debate System**: AI-powered discussions between different investor personalities
- **Consensus Building**: Algorithmic consensus scoring based on multiple persona analyses

## External Dependencies

### Core Dependencies
- **Database**: PostgreSQL via Neon Database serverless platform
- **AI Services**: OpenAI API for GPT-powered analysis and content generation
- **Financial Data**: Multiple provider support (Alpha Vantage, Financial Datasets API)
- **News API**: NewsAPI.org for market and stock-specific news feeds

### Development Tools
- **Build System**: Vite with React plugin and TypeScript support
- **Code Quality**: TypeScript for type safety and better developer experience
- **UI Components**: Radix UI primitives with Shadcn/ui component library
- **Styling**: Tailwind CSS with PostCSS for processing

### Authentication & Session Management
- **Session Storage**: PostgreSQL-based session storage via `connect-pg-simple`
- **Security**: Express session middleware for user state management

### Monitoring & Development
- **Error Handling**: Replit-specific error overlay and development tools
- **Logging**: Custom request/response logging with performance timing
- **Development**: Replit-integrated development environment with cartographer and dev banner plugins

### Data Validation
- **Schema Validation**: Zod for runtime type checking and API validation
- **Database Validation**: Drizzle-Zod integration for database schema validation
- **Form Handling**: React Hook Form with Hookform resolvers for client-side validation