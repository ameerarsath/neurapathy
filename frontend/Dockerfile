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

# Build arguments
ARG REACT_APP_API_URL=http://localhost:8080/api
ARG REACT_APP_ML_API_URL=http://localhost:8000
ARG REACT_APP_WS_URL=ws://localhost:8080
ARG REACT_APP_VERSION=1.0.0
ARG REACT_APP_BUILD_TIME
ARG REACT_APP_COMMIT_HASH

# Set environment variables
ENV REACT_APP_API_URL=$REACT_APP_API_URL
ENV REACT_APP_ML_API_URL=$REACT_APP_ML_API_URL
ENV REACT_APP_WS_URL=$REACT_APP_WS_URL
ENV REACT_APP_VERSION=$REACT_APP_VERSION
ENV REACT_APP_BUILD_TIME=$REACT_APP_BUILD_TIME
ENV REACT_APP_COMMIT_HASH=$REACT_APP_COMMIT_HASH
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