package com.smartshoe.api.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.web.socket.config.annotation.EnableWebSocket;
import org.springframework.web.socket.config.annotation.WebSocketConfigurer;
import org.springframework.web.socket.config.annotation.WebSocketHandlerRegistry;
import org.springframework.web.socket.server.support.HttpSessionHandshakeInterceptor;
import com.smartshoe.api.websocket.SmartShoeWebSocketHandler;

/**
 * WebSocket Configuration for Smart Shoe Real-time Communication
 * 
 * This configuration enables WebSocket connections for real-time data streaming
 * from smart shoe devices, including sensor data, alerts, and notifications.
 */
@Configuration
@EnableWebSocket
public class WebSocketConfig implements WebSocketConfigurer {

    private final SmartShoeWebSocketHandler webSocketHandler;

    public WebSocketConfig(SmartShoeWebSocketHandler webSocketHandler) {
        this.webSocketHandler = webSocketHandler;
    }

    @Override
    public void registerWebSocketHandlers(WebSocketHandlerRegistry registry) {
        // Register WebSocket endpoint for real-time communication
        registry.addHandler(webSocketHandler, "/ws")
                .setAllowedOrigins("*") // Allow all origins for development
                .addInterceptors(new HttpSessionHandshakeInterceptor());
        
        // Register alternative endpoints for different use cases
        registry.addHandler(webSocketHandler, "/websocket")
                .setAllowedOrigins("*")
                .addInterceptors(new HttpSessionHandshakeInterceptor());
                
        // Register SockJS fallback for browsers that don't support WebSocket
        registry.addHandler(webSocketHandler, "/ws")
                .setAllowedOrigins("*")
                .withSockJS();
    }
}