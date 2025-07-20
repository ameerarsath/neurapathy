# Multi-stage build for production-ready React frontend
FROM node:18-alpine as build

# Set working directory
WORKDIR /app

# Install dependencies for building
RUN apk add --no-cache \
    build-base \
    python3 \
    make \
    g++

# Copy package files
COPY package*.json ./
COPY yarn.lock* ./

# Install dependencies
RUN npm ci --only=production --frozen-lockfile

# Copy source code
COPY . .

# Build arguments (Updated for Vite)
ARG VITE_API_BASE_URL=http://13.201.120.175/api
ARG VITE_ML_API_BASE_URL=http://13.201.120.175/api/ml
ARG VITE_WS_URL=ws://13.201.120.175/ws
ARG VITE_ENV=production
ARG VITE_VERSION=1.0.0
ARG VITE_BUILD_TIME
ARG VITE_COMMIT_HASH

# Set environment variables (Updated for Vite)
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
ENV VITE_ML_API_BASE_URL=$VITE_ML_API_BASE_URL
ENV VITE_WS_URL=$VITE_WS_URL
ENV VITE_ENV=$VITE_ENV
ENV VITE_VERSION=$VITE_VERSION
ENV VITE_BUILD_TIME=$VITE_BUILD_TIME
ENV VITE_COMMIT_HASH=$VITE_COMMIT_HASH
ENV NODE_ENV=production

# Build the application
RUN npm run build

# Production stage
FROM nginx:alpine

# Install additional tools
RUN apk add --no-cache \
    curl \
    jq \
    tzdata

# Copy built application
COPY --from=build /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf
COPY nginx.default.conf /etc/nginx/conf.d/default.conf

# Copy SSL certificates (if using HTTPS)
COPY ssl/ /etc/nginx/ssl/

# Create nginx user and set permissions
RUN addgroup -g 1001 -S nginx && \
    adduser -S -D -H -u 1001 -h /var/cache/nginx -s /sbin/nologin -G nginx -g nginx nginx && \
    chown -R nginx:nginx /usr/share/nginx/html && \
    chown -R nginx:nginx /var/cache/nginx && \
    chown -R nginx:nginx /var/log/nginx && \
    chown -R nginx:nginx /etc/nginx/conf.d

# Switch to non-root user
USER nginx

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

# Labels for better container management
LABEL maintainer="Smart Shoe Team <dev@smartshoe.com>"
LABEL description="Production-ready frontend for Diabetic Smart Shoe Monitoring Platform"
LABEL version="1.0.0"

# Start nginx
CMD ["nginx", "-g", "daemon off;"]