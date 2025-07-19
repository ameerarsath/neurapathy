# 🦶 Diabetic Smart Shoe Frontend

A production-ready React frontend for the Diabetic Smart Shoe Monitoring Platform, providing healthcare professionals and patients with real-time neuropathy monitoring, ML-powered insights, and comprehensive health analytics.

## 🌟 Features

### Core Features
- **🔐 Advanced Authentication** - JWT-based auth with 2FA support
- **📊 Real-time Dashboard** - Live monitoring with WebSocket updates
- **🧠 ML Integration** - Real-time neuropathy progression predictions
- **📱 Progressive Web App** - Offline capabilities and mobile optimization
- **🔔 Smart Notifications** - Context-aware alerts and push notifications
- **🎨 Adaptive Themes** - Light, dark, and medical theme support
- **♿ Accessibility** - WCAG 2.1 AA compliant interface
- **🌍 Internationalization** - Multi-language support

### Medical Features
- **🏥 Patient Management** - Comprehensive patient profiles and history
- **🔬 Test Administration** - Guided neuropathy testing interface
- **📈 Analytics & Reporting** - Advanced health metrics visualization
- **⚕️ Clinical Decision Support** - ML-powered risk assessment
- **🔌 Device Integration** - Smart shoe device management
- **📋 Compliance Tracking** - HIPAA-compliant data handling

### Technical Features
- **⚡ Performance Optimized** - Code splitting and lazy loading
- **🔄 Real-time Updates** - WebSocket integration for live data
- **📱 Responsive Design** - Mobile-first, cross-platform compatibility
- **🛡️ Security Hardened** - CSP, XSS protection, and data encryption
- **🔍 Monitoring & Analytics** - Performance tracking and error reporting
- **🧪 Testing Suite** - Comprehensive unit and integration tests

## 🚀 Technology Stack

### Frontend Core
- **React 18** - Modern React with Suspense and Concurrent Features
- **TypeScript** - Type-safe development
- **Vite 5** - Lightning-fast build tool
- **Tailwind CSS** - Utility-first styling
- **Framer Motion** - Smooth animations and transitions

### State Management & Data
- **React Query** - Server state management with caching
- **Zustand** - Lightweight client state management
- **Axios** - HTTP client with interceptors
- **Socket.IO** - Real-time bidirectional communication

### UI & Visualization
- **Recharts** - Responsive chart library
- **Chart.js** - Advanced data visualization
- **Lucide React** - Beautiful icon library
- **React Hook Form** - Performant form handling
- **React Select** - Accessible select components

### Development & Build
- **ESLint** - Code linting and formatting
- **Prettier** - Code formatting
- **Husky** - Git hooks for quality gates
- **Vitest** - Unit testing framework
- **MSW** - API mocking for testing

## 🏗️ Installation & Setup

### Prerequisites
- Node.js 18+ and npm 9+
- Git for version control

### Quick Start
```bash
# Clone the repository
git clone https://github.com/yourorg/diabetic-smart-shoe.git
cd diabetic-smart-shoe/frontend

# Install dependencies
npm install

# Copy environment configuration
cp .env.example .env.local

# Start development server
npm run dev
```

### Environment Configuration
Edit `.env.local` with your configuration:

```env
# API Configuration
VITE_API_BASE_URL=http://localhost:8080/api
VITE_ML_API_BASE_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8080

# Feature Flags
VITE_ENABLE_PWA=true
VITE_ENABLE_PUSH_NOTIFICATIONS=true
VITE_ENABLE_OFFLINE_MODE=true

# Security
VITE_ENCRYPTION_KEY=your-encryption-key-here
```

## 📜 Available Scripts

### Development
```bash
npm run dev          # Start development server
npm run dev:host     # Start with network access
npm run dev:debug    # Start with debugging enabled
```

### Building
```bash
npm run build        # Production build
npm run build:prod   # Production build with optimizations
npm run preview      # Preview production build
npm run analyze      # Bundle size analysis
```

### Quality & Testing
```bash
npm run lint         # Run ESLint
npm run lint:fix     # Fix linting issues
npm run format       # Format code with Prettier
npm run test         # Run unit tests
npm run test:ui      # Run tests with UI
npm run test:coverage # Run tests with coverage
npm run type-check   # TypeScript type checking
```

## 🏛️ Architecture

### Project Structure
```
src/
├── components/          # Reusable UI components
│   ├── common/         # Generic components
│   ├── forms/          # Form components
│   ├── layout/         # Layout components
│   └── charts/         # Chart components
├── pages/              # Page components
│   ├── Auth/           # Authentication pages
│   ├── Dashboard/      # Dashboard pages
│   ├── Patients/       # Patient management
│   ├── Analytics/      # Analytics pages
│   └── Settings/       # Settings pages
├── contexts/           # React context providers
├── hooks/              # Custom React hooks
├── services/           # API services
├── utils/              # Utility functions
├── types/              # TypeScript definitions
├── constants/          # Application constants
├── assets/             # Static assets
└── styles/             # Global styles
```

### Component Architecture
- **Atomic Design** - Organized component hierarchy
- **Compound Components** - Complex UI patterns
- **Render Props** - Flexible component composition
- **Higher-Order Components** - Cross-cutting concerns

### State Management
- **Server State** - React Query for API data
- **Client State** - Zustand for UI state
- **Context** - Theme, auth, and global state
- **Local State** - useState and useReducer

## 🔧 Development Guide

### Code Standards
- **ESLint Configuration** - Airbnb + custom rules
- **Prettier** - Consistent code formatting
- **TypeScript** - Strict type checking
- **Commit Convention** - Conventional commits

### Component Development
```jsx
// Example component structure
import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useQuery } from 'react-query'

interface ComponentProps {
  id: string
  className?: string
  onUpdate?: (data: any) => void
}

export const Component: React.FC<ComponentProps> = ({
  id,
  className,
  onUpdate
}) => {
  // Component logic here
  return (
    <motion.div
      className={className}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
    >
      {/* Component JSX */}
    </motion.div>
  )
}
```

### API Integration
```typescript
// API service example
import { api } from '@services/api'

export const usePatientData = (patientId: string) => {
  return useQuery(['patient', patientId], () =>
    api.patient.getPatientById(patientId)
  )
}
```

## 🚀 Deployment

### Production Build
```bash
# Build for production
npm run build:prod

# Test production build
npm run preview
```

### Docker Deployment
```bash
# Build Docker image
docker build -t diabetic-smart-shoe-frontend .

# Run container
docker run -p 3000:3000 diabetic-smart-shoe-frontend
```

### Environment-Specific Builds
```bash
# Staging environment
npm run build:staging

# Production environment
npm run build:production
```

## 📊 Monitoring & Analytics

### Performance Monitoring
- **Web Vitals** - Core web vitals tracking
- **Bundle Analysis** - Code splitting optimization
- **Memory Usage** - Memory leak detection
- **Network Requests** - API performance monitoring

### Error Tracking
- **Error Boundaries** - React error handling
- **Sentry Integration** - Production error tracking
- **Console Monitoring** - Development debugging

### User Analytics
- **Google Analytics** - User behavior tracking
- **Hotjar** - User session recording
- **Custom Events** - Feature usage analytics

## 🛡️ Security

### Authentication & Authorization
- **JWT Tokens** - Secure token-based auth
- **Role-based Access** - Granular permissions
- **Session Management** - Secure session handling
- **2FA Support** - Two-factor authentication

### Data Security
- **Encryption** - Client-side data encryption
- **HTTPS Only** - Secure communication
- **CSP Headers** - Content Security Policy
- **XSS Protection** - Cross-site scripting prevention

### HIPAA Compliance
- **Data Minimization** - Collect only necessary data
- **Access Logging** - Audit trail for all access
- **Secure Storage** - Encrypted data storage
- **User Consent** - Clear consent mechanisms

## 🧪 Testing

### Testing Strategy
- **Unit Tests** - Component and utility testing
- **Integration Tests** - API integration testing
- **E2E Tests** - User workflow testing
- **Visual Tests** - UI regression testing

### Test Configuration
```javascript
// vitest.config.js
export default {
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.js'],
    coverage: {
      reporter: ['text', 'json', 'html'],
      threshold: {
        global: {
          branches: 80,
          functions: 80,
          lines: 80,
          statements: 80
        }
      }
    }
  }
}
```

## 📱 Progressive Web App

### PWA Features
- **Service Worker** - Offline caching
- **App Manifest** - Native app experience
- **Push Notifications** - Real-time alerts
- **Background Sync** - Offline data sync

### Installation
The app can be installed on supported devices:
1. Visit the application in a supported browser
2. Click "Install" when prompted
3. Or use browser's "Add to Home Screen" option

## 🌍 Internationalization

### Supported Languages
- English (en) - Default
- Spanish (es) - Healthcare professionals
- French (fr) - Canadian market

### Adding New Languages
```bash
# Generate translation keys
npm run i18n:extract

# Add new language file
touch src/locales/de.json

# Update configuration
# Edit src/i18n/config.ts
```

## 🤝 Contributing

### Development Workflow
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests: `npm run test`
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push to branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Code Review Process
- All PRs require review from 2 team members
- Automated tests must pass
- Code coverage must be maintained
- Security scan must pass

## 📚 Documentation

### API Documentation
- **OpenAPI Spec** - Interactive API documentation
- **Postman Collection** - API testing collection
- **SDK Documentation** - Client library docs

### Component Documentation
- **Storybook** - Component library documentation
- **Props Documentation** - TypeScript-generated docs
- **Usage Examples** - Real-world usage patterns

## 🚨 Troubleshooting

### Common Issues

**Build Issues**
```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Clear Vite cache
rm -rf .vite
npm run dev
```

**Performance Issues**
```bash
# Analyze bundle size
npm run analyze

# Check for memory leaks
npm run dev:debug
```

**API Connection Issues**
```bash
# Verify API endpoint
curl http://localhost:8080/api/health

# Check CORS configuration
```

## 📞 Support

### Getting Help
- **GitHub Issues** - Bug reports and feature requests
- **Documentation** - Comprehensive guides and API docs
- **Discord** - Community support and discussions
- **Email** - Technical support at support@smartshoe.com

### Emergency Contacts
- **Security Issues** - security@smartshoe.com
- **Critical Bugs** - critical@smartshoe.com
- **Medical Emergencies** - This is not a medical emergency system

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Healthcare Professionals** - For clinical guidance and testing
- **Patients** - For valuable feedback and insights
- **Open Source Community** - For the amazing tools and libraries
- **Security Researchers** - For responsible disclosure of vulnerabilities

---

**Built with ❤️ by the Smart Shoe Team**

For more information, visit our [website](https://smartshoe.com) or contact us at [info@smartshoe.com](mailto:info@smartshoe.com).
