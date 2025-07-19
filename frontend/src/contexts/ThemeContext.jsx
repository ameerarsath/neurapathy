import React, { createContext, useContext, useEffect, useState } from 'react'

// Theme configuration
const themes = {
  light: {
    primary: '#3b82f6',
    secondary: '#64748b',
    accent: '#10b981',
    background: '#ffffff',
    surface: '#f8fafc',
    text: '#1e293b',
    textSecondary: '#64748b',
    border: '#e2e8f0',
    error: '#ef4444',
    warning: '#f59e0b',
    success: '#10b981',
    info: '#3b82f6',
    cardBackground: '#ffffff',
    sidebarBackground: '#f8fafc',
    headerBackground: '#ffffff',
    shadow: 'rgba(0, 0, 0, 0.1)'
  },
  dark: {
    primary: '#3b82f6',
    secondary: '#64748b',
    accent: '#10b981',
    background: '#0f172a',
    surface: '#1e293b',
    text: '#f1f5f9',
    textSecondary: '#94a3b8',
    border: '#334155',
    error: '#ef4444',
    warning: '#f59e0b',
    success: '#10b981',
    info: '#3b82f6',
    cardBackground: '#1e293b',
    sidebarBackground: '#0f172a',
    headerBackground: '#1e293b',
    shadow: 'rgba(0, 0, 0, 0.3)'
  },
  medical: {
    primary: '#059669',
    secondary: '#64748b',
    accent: '#0ea5e9',
    background: '#f0fdf4',
    surface: '#ffffff',
    text: '#064e3b',
    textSecondary: '#047857',
    border: '#bbf7d0',
    error: '#dc2626',
    warning: '#d97706',
    success: '#059669',
    info: '#0ea5e9',
    cardBackground: '#ffffff',
    sidebarBackground: '#f0fdf4',
    headerBackground: '#ffffff',
    shadow: 'rgba(5, 150, 105, 0.1)'
  }
}

// Create context
const ThemeContext = createContext()

// Custom hook to use theme
export const useTheme = () => {
  const context = useContext(ThemeContext)
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider')
  }
  return context
}

// Theme provider component
export const ThemeProvider = ({ children }) => {
  const [currentTheme, setCurrentTheme] = useState('light')
  const [systemPreference, setSystemPreference] = useState('light')

  // Detect system theme preference
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
    setSystemPreference(mediaQuery.matches ? 'dark' : 'light')

    const handleChange = (e) => {
      setSystemPreference(e.matches ? 'dark' : 'light')
    }

    mediaQuery.addEventListener('change', handleChange)
    return () => mediaQuery.removeEventListener('change', handleChange)
  }, [])

  // Load theme preference from localStorage
  useEffect(() => {
    const savedTheme = localStorage.getItem('theme')
    if (savedTheme && themes[savedTheme]) {
      setCurrentTheme(savedTheme)
    } else {
      setCurrentTheme(systemPreference)
    }
  }, [systemPreference])

  // Apply theme to document
  useEffect(() => {
    const theme = themes[currentTheme]
    const root = document.documentElement
    
    // Set CSS custom properties
    Object.entries(theme).forEach(([key, value]) => {
      root.style.setProperty(`--color-${key}`, value)
    })

    // Set theme class on body
    document.body.className = `theme-${currentTheme}`
    
    // Set meta theme color for mobile browsers
    const metaThemeColor = document.querySelector('meta[name="theme-color"]')
    if (metaThemeColor) {
      metaThemeColor.setAttribute('content', theme.primary)
    }
  }, [currentTheme])

  const switchTheme = (themeName) => {
    if (themes[themeName]) {
      setCurrentTheme(themeName)
      localStorage.setItem('theme', themeName)
    }
  }

  const toggleTheme = () => {
    const newTheme = currentTheme === 'light' ? 'dark' : 'light'
    switchTheme(newTheme)
  }

  const getThemeColors = () => themes[currentTheme]

  const isDarkMode = currentTheme === 'dark'

  const value = {
    currentTheme,
    themes,
    switchTheme,
    toggleTheme,
    getThemeColors,
    isDarkMode,
    systemPreference
  }

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  )
}

export default ThemeProvider