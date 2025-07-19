# WebSocket Setup Instructions

## Current Status
WebSocket connections are **disabled by default** in development mode to prevent connection errors when no WebSocket server is running.

## How to Enable WebSocket

### Method 1: Browser Storage (Temporary)
Open browser console and run:
```javascript
localStorage.setItem('enableWebSocket', 'true')
```
Then refresh the page.

### Method 2: Environment Variable (Permanent)
Add to your `.env` file:
```
VITE_APP_ENVIRONMENT=production
```

## To Add WebSocket Backend Support

1. Add WebSocket dependency to `backend/pom.xml`:
```xml
<dependency>
    <groupId>org.springframework</groupId>
    <artifactId>spring-websocket</artifactId>
</dependency>
```

2. Create WebSocket configuration and handlers in the backend.

## Current Behavior
- **Development Mode**: WebSocket disabled, no connection errors
- **Production Mode**: WebSocket enabled automatically
- **Manual Enable**: Use localStorage flag to enable in development

## Testing WebSocket
When enabled, you can test WebSocket functionality by:
1. Opening browser console
2. Looking for WebSocket connection messages
3. Using the notification system which integrates with WebSocket events

## Features Affected by WebSocket
- Real-time notifications
- Live device data updates
- Test result notifications
- Medical alerts
- System status updates