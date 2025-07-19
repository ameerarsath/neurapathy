# ====================================================================
# PRODUCTION-LEVEL DOCKERFILE FOR SMART SHOE API
# Multi-stage build with security, monitoring, and performance optimization
# ====================================================================

# ====================================================================
# STAGE 1: BUILD STAGE
# ====================================================================
FROM maven:3.9.6-eclipse-temurin-17 AS builder

# Set metadata
LABEL stage=builder
LABEL maintainer="Smart Shoe Development Team"

# Set working directory
WORKDIR /app

# Copy Maven wrapper files first for better layer caching
COPY mvnw .
COPY .mvn .mvn

# Copy Maven configuration files (for better caching)
COPY pom.xml .

# Download dependencies first (this step will be cached if pom.xml doesn't change)
RUN mvn dependency:go-offline -B --no-transfer-progress

# Copy source code
COPY src ./src

# Build the application with optimized settings
RUN mvn clean package -DskipTests -B --no-transfer-progress \
    -Dspring.profiles.active=production \
    -Djava.awt.headless=true

# ====================================================================
# STAGE 2: PRODUCTION RUNTIME STAGE
# ====================================================================
FROM eclipse-temurin:17-jre-alpine AS production

# Set comprehensive metadata
LABEL maintainer="Smart Shoe Development Team"
LABEL description="Smart Shoe API - Production-Ready Diabetic Neuropathy Monitoring Platform"
LABEL version="3.0.0"
LABEL vendor="Smart Shoe Inc."
LABEL org.opencontainers.image.title="Smart Shoe API"
LABEL org.opencontainers.image.description="HIPAA-compliant diabetic neuropathy monitoring platform"
LABEL org.opencontainers.image.version="3.0.0"
LABEL org.opencontainers.image.vendor="Smart Shoe Inc."

# Install production packages and security updates
RUN apk update && apk upgrade && \
    apk add --no-cache \
    curl \
    wget \
    tzdata \
    ca-certificates \
    bash \
    dumb-init \
    && update-ca-certificates \
    && ln -sf /usr/share/zoneinfo/UTC /etc/localtime \
    && rm -rf /var/cache/apk/*

# Create non-root user with specific UID/GID for security
RUN addgroup -g 1001 -S smartshoe && \
    adduser -S smartshoe -u 1001 -G smartshoe -h /app -s /bin/bash

# Set working directory
WORKDIR /app

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs /app/data /app/config /app/temp && \
    chown -R smartshoe:smartshoe /app

# Copy built application from builder stage
COPY --from=builder --chown=smartshoe:smartshoe /app/target/api-*.jar app.jar

# Copy production configuration files
COPY --chown=smartshoe:smartshoe src/main/resources/application.yml ./config/

# Set file permissions for security
RUN chmod 755 /app && \
    chmod 644 app.jar && \
    chmod -R 755 /app/logs /app/data /app/config /app/temp

# Switch to non-root user
USER smartshoe

# Expose application port
EXPOSE 8080

# Production-optimized health check
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=5 \
    CMD wget --no-verbose --tries=1 --spider --timeout=10 \
        http://localhost:8080/actuator/health || exit 1

# Production JVM optimization
ENV JAVA_OPTS="-server \
    -Xmx512m \
    -Xms256m \
    -XX:+UseG1GC \
    -XX:+UseStringDeduplication \
    -XX:MaxGCPauseMillis=200 \
    -XX:+HeapDumpOnOutOfMemoryError \
    -XX:HeapDumpPath=/app/logs/ \
    -XX:+UseContainerSupport \
    -XX:MaxRAMPercentage=80.0 \
    -Djava.security.egd=file:/dev/./urandom \
    -Djava.awt.headless=true \
    -Dfile.encoding=UTF-8 \
    -Duser.timezone=UTC"

# Production Spring Boot configuration
ENV SPRING_PROFILES_ACTIVE=production
ENV LOGGING_LEVEL_ROOT=INFO
ENV LOGGING_LEVEL_SMARTSHOE=DEBUG
ENV SERVER_PORT=8080
ENV MANAGEMENT_ENDPOINTS_WEB_EXPOSURE_INCLUDE=health,info,metrics
ENV MANAGEMENT_ENDPOINT_HEALTH_SHOW_DETAILS=when_authorized

# Security environment variables
ENV JAVA_SECURITY_PROPERTIES=/app/config/java.security
ENV SPRING_SECURITY_REQUIRE_SSL=false

# Use dumb-init for proper signal handling
ENTRYPOINT ["dumb-init", "--"]

# Production startup command with comprehensive error handling
CMD ["sh", "-c", "exec java $JAVA_OPTS -jar app.jar"]