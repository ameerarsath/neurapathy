import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Download, X, Smartphone, Monitor } from 'lucide-react'

const PWAInstallPrompt = () => {
  const [deferredPrompt, setDeferredPrompt] = useState(null)
  const [showInstallPrompt, setShowInstallPrompt] = useState(false)
  const [isInstalled, setIsInstalled] = useState(false)
  const [installSource, setInstallSource] = useState(null)

  useEffect(() => {
    // Check if app is already installed
    const checkIfInstalled = () => {
      // Check for standalone mode (PWA installed)
      if (window.matchMedia('(display-mode: standalone)').matches) {
        setIsInstalled(true)
        return
      }

      // Check for navigator.standalone (iOS Safari)
      if (window.navigator.standalone) {
        setIsInstalled(true)
        return
      }

      // Check for document.referrer (Android)
      if (document.referrer.includes('android-app://')) {
        setIsInstalled(true)
        return
      }
    }

    checkIfInstalled()

    // Listen for beforeinstallprompt event
    const handleBeforeInstallPrompt = (e) => {
      console.log('beforeinstallprompt event fired')
      e.preventDefault()
      setDeferredPrompt(e)
      
      // Show install prompt after a delay if not already installed
      if (!isInstalled) {
        setTimeout(() => {
          setShowInstallPrompt(true)
        }, 5000) // Wait 5 seconds before showing prompt
      }
    }

    // Listen for appinstalled event
    const handleAppInstalled = (e) => {
      console.log('PWA was installed')
      setIsInstalled(true)
      setShowInstallPrompt(false)
      setDeferredPrompt(null)
    }

    window.addEventListener('beforeinstallprompt', handleBeforeInstallPrompt)
    window.addEventListener('appinstalled', handleAppInstalled)

    return () => {
      window.removeEventListener('beforeinstallprompt', handleBeforeInstallPrompt)
      window.removeEventListener('appinstalled', handleAppInstalled)
    }
  }, [isInstalled])

  // Handle install button click
  const handleInstallClick = async () => {
    if (!deferredPrompt) {
      // Fallback for browsers that don't support beforeinstallprompt
      showManualInstallInstructions()
      return
    }

    try {
      // Show the install prompt
      deferredPrompt.prompt()
      
      // Wait for the user to respond to the prompt
      const { outcome } = await deferredPrompt.userChoice
      
      console.log(`User response to install prompt: ${outcome}`)
      
      if (outcome === 'accepted') {
        setIsInstalled(true)
      }
      
      // Clear the deferred prompt
      setDeferredPrompt(null)
      setShowInstallPrompt(false)
    } catch (error) {
      console.error('Error during PWA installation:', error)
      showManualInstallInstructions()
    }
  }

  // Show manual install instructions
  const showManualInstallInstructions = () => {
    const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent)
    const isAndroid = /Android/.test(navigator.userAgent)
    
    let instructions = ''
    
    if (isIOS) {
      instructions = 'Tap the Share button and select "Add to Home Screen"'
      setInstallSource('ios')
    } else if (isAndroid) {
      instructions = 'Tap the menu (⋮) and select "Add to Home Screen" or "Install App"'
      setInstallSource('android')
    } else {
      instructions = 'Look for the install icon in your browser address bar or menu'
      setInstallSource('desktop')
    }
    
    // You could show a modal with instructions here
    alert(instructions)
  }

  // Handle dismiss
  const handleDismiss = () => {
    setShowInstallPrompt(false)
    
    // Don't show again for this session
    sessionStorage.setItem('pwa-install-dismissed', 'true')
  }

  // Check if user has already dismissed for this session
  useEffect(() => {
    const dismissed = sessionStorage.getItem('pwa-install-dismissed')
    if (dismissed) {
      setShowInstallPrompt(false)
    }
  }, [])

  // Don't show if already installed
  if (isInstalled) {
    return null
  }

  return (
    <AnimatePresence>
      {showInstallPrompt && (
        <motion.div
          initial={{ opacity: 0, y: 100 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 100 }}
          className="fixed bottom-4 left-4 right-4 md:left-auto md:right-4 md:max-w-sm z-50"
        >
          <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg p-4">
            {/* Header */}
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 bg-blue-100 dark:bg-blue-900 rounded-lg flex items-center justify-center">
                  <Smartphone className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                </div>
                <h3 className="font-semibold text-gray-900 dark:text-white text-sm">
                  Install Smart Shoe App
                </h3>
              </div>
              <button
                onClick={handleDismiss}
                className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Content */}
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
              Get quick access to your diabetic monitoring dashboard. Install the app for a better experience.
            </p>

            {/* Features */}
            <div className="space-y-2 mb-4">
              <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
                <div className="w-1 h-1 bg-green-500 rounded-full"></div>
                <span>Offline access to your data</span>
              </div>
              <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
                <div className="w-1 h-1 bg-green-500 rounded-full"></div>
                <span>Push notifications for alerts</span>
              </div>
              <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
                <div className="w-1 h-1 bg-green-500 rounded-full"></div>
                <span>Faster loading times</span>
              </div>
            </div>

            {/* Actions */}
            <div className="flex gap-2">
              <button
                onClick={handleInstallClick}
                className="flex-1 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium py-2 px-3 rounded-lg transition-colors duration-200 flex items-center justify-center gap-2"
              >
                <Download className="w-4 h-4" />
                Install
              </button>
              <button
                onClick={handleDismiss}
                className="px-3 py-2 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 transition-colors"
              >
                Later
              </button>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

export default PWAInstallPrompt