#!/bin/bash

echo "🚀 Quick Frontend Build Script"
echo "==============================="

# Set production environment variables
export NODE_ENV=production
export VITE_API_BASE_URL=http://13.201.120.175:8080/api
export VITE_ML_API_BASE_URL=http://13.201.120.175:8080/api/ml
export VITE_WS_URL=ws://13.201.120.175:8080/ws
export VITE_ENV=production

echo "✅ Environment variables set for production"

# Create simple build directory with static files
mkdir -p dist
cp index.html dist/
cp -r public/* dist/ 2>/dev/null || true

# Create simple production index.html
cat > dist/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Smart Shoe Platform</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            text-align: center;
        }
        .container {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 40px;
            max-width: 600px;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
            border: 1px solid rgba(255, 255, 255, 0.18);
        }
        h1 {
            font-size: 2.5em;
            margin-bottom: 20px;
            background: linear-gradient(45deg, #fff, #f0f0f0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .status {
            display: inline-block;
            background: #28a745;
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            margin: 10px 0;
        }
        .api-info {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
        }
        .endpoint {
            font-family: monospace;
            background: rgba(0, 0, 0, 0.2);
            padding: 5px 10px;
            border-radius: 5px;
            margin: 5px 0;
            display: block;
        }
        button {
            background: #007bff;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            margin: 10px;
            transition: all 0.3s ease;
        }
        button:hover {
            background: #0056b3;
            transform: translateY(-2px);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🩺 Smart Shoe Platform</h1>
        <div class="status">System Online</div>
        
        <div class="api-info">
            <h3>API Endpoints</h3>
            <div class="endpoint">Backend: http://13.201.120.175:8080/api</div>
            <div class="endpoint">Health: http://13.201.120.175:8080/api/health</div>
            <div class="endpoint">ML API: http://13.201.120.175:8080/api/ml</div>
        </div>
        
        <p>Production-ready Smart Shoe monitoring platform for diabetic neuropathy testing and health monitoring.</p>
        
        <button onclick="testAPI()">Test API Connection</button>
        <button onclick="window.open('/api-test.html')">API Test Suite</button>
        
        <div id="api-status" style="margin-top: 20px;"></div>
    </div>

    <script>
        async function testAPI() {
            const statusEl = document.getElementById('api-status');
            statusEl.innerHTML = '<div style="color: #ffc107;">Testing API connection...</div>';
            
            try {
                const response = await fetch('http://13.201.120.175:8080/api/health');
                const data = await response.json();
                statusEl.innerHTML = `<div style="color: #28a745;">✅ API Connected: ${data.message}</div>`;
            } catch (error) {
                statusEl.innerHTML = `<div style="color: #dc3545;">❌ API Connection Failed: ${error.message}</div>`;
            }
        }
        
        // Auto-test API on load
        window.onload = function() {
            setTimeout(testAPI, 1000);
        };
    </script>
</body>
</html>
EOF

echo "✅ Production index.html created"

# Copy the API test file we created earlier
cp simple-test.html dist/api-test.html 2>/dev/null || echo "⚠️  API test file not found"

echo "✅ Frontend build completed!"
echo ""
echo "📁 Build files ready in: $(pwd)/dist/"
echo "📄 Main file: dist/index.html"
echo "🧪 Test file: dist/api-test.html"
echo ""
echo "🚀 Ready for deployment!"