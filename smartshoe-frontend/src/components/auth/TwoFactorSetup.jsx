import React, { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { smartShoeAPI } from '../../services/api'
import LoadingSpinner from '../common/LoadingSpinner'
import { Shield, Eye, EyeOff, Key, Copy, CheckCircle } from 'lucide-react'
import toast from 'react-hot-toast'

const TwoFactorSetup = () => {
  const [showSetup, setShowSetup] = useState(false)
  const [verificationCode, setVerificationCode] = useState('')
  const [setupData, setSetupData] = useState(null)
  const [showBackupCodes, setShowBackupCodes] = useState(false)
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const queryClient = useQueryClient()

  // Get 2FA status
  const { data: twoFactorStatus, isLoading: statusLoading } = useQuery({
    queryKey: ['twoFactorStatus'],
    queryFn: () => smartShoeAPI.auth.getTwoFactorStatus(),
    select: data => data.data
  })

  // Enable 2FA mutation
  const enableTwoFactorMutation = useMutation({
    mutationFn: () => smartShoeAPI.auth.enableTwoFactor(),
    onSuccess: (data) => {
      setSetupData(data.data)
      setShowSetup(true)
      toast.success('2FA setup initiated. Please scan the QR code with your authenticator app.')
    },
    onError: (error) => {
      toast.error(error.response?.data?.message || 'Failed to enable 2FA')
    }
  })

  // Verify 2FA setup mutation
  const verifySetupMutation = useMutation({
    mutationFn: (totpCode) => smartShoeAPI.auth.verifyTwoFactorSetup({ totpCode }),
    onSuccess: () => {
      toast.success('Two-factor authentication enabled successfully!')
      setShowSetup(false)
      setSetupData(null)
      setVerificationCode('')
      setShowBackupCodes(true)
      queryClient.invalidateQueries(['twoFactorStatus'])
    },
    onError: (error) => {
      toast.error(error.response?.data?.message || 'Invalid verification code')
    }
  })

  // Disable 2FA mutation
  const disableTwoFactorMutation = useMutation({
    mutationFn: (password) => smartShoeAPI.auth.disableTwoFactor({ password }),
    onSuccess: () => {
      toast.success('Two-factor authentication disabled')
      setPassword('')
      queryClient.invalidateQueries(['twoFactorStatus'])
    },
    onError: (error) => {
      toast.error(error.response?.data?.message || 'Failed to disable 2FA')
    }
  })

  // Regenerate backup codes mutation
  const regenerateBackupCodesMutation = useMutation({
    mutationFn: () => smartShoeAPI.auth.regenerateBackupCodes(),
    onSuccess: (data) => {
      setSetupData(prev => ({ ...prev, backupCodes: data.data.backupCodes }))
      setShowBackupCodes(true)
      toast.success('New backup codes generated')
    },
    onError: (error) => {
      toast.error(error.response?.data?.message || 'Failed to regenerate backup codes')
    }
  })

  const handleEnableTwoFactor = () => {
    enableTwoFactorMutation.mutate()
  }

  const handleVerifySetup = (e) => {
    e.preventDefault()
    if (verificationCode.length === 6) {
      verifySetupMutation.mutate(verificationCode)
    }
  }

  const handleDisableTwoFactor = (e) => {
    e.preventDefault()
    disableTwoFactorMutation.mutate(password)
  }

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text)
    toast.success('Copied to clipboard')
  }

  const copyBackupCodes = () => {
    const codes = setupData.backupCodes.join('\n')
    navigator.clipboard.writeText(codes)
    toast.success('Backup codes copied to clipboard')
  }

  if (statusLoading) {
    return (
      <div className="flex items-center justify-center min-h-[200px]">
        <LoadingSpinner size="lg" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* 2FA Status */}
      <div className="medical-card">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Shield className={`h-6 w-6 ${twoFactorStatus?.twoFactorEnabled ? 'text-success' : 'text-neutral-400'}`} />
            <div>
              <h3 className="text-lg font-medium text-neutral-900">
                Two-Factor Authentication
              </h3>
              <p className="text-sm text-neutral-600">
                {twoFactorStatus?.twoFactorEnabled 
                  ? 'Your account is protected with 2FA'
                  : 'Add an extra layer of security to your account'
                }
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <span className={`status-indicator ${twoFactorStatus?.twoFactorEnabled ? 'status-normal' : 'status-warning'}`}>
              {twoFactorStatus?.twoFactorEnabled ? 'Enabled' : 'Disabled'}
            </span>
            {twoFactorStatus?.twoFactorEnabled ? (
              <div className="flex space-x-2">
                <button
                  onClick={() => regenerateBackupCodesMutation.mutate()}
                  disabled={regenerateBackupCodesMutation.isLoading}
                  className="inline-flex items-center px-4 py-2 border border-neutral-300 text-sm font-medium rounded-md text-neutral-700 bg-white hover:bg-neutral-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {regenerateBackupCodesMutation.isLoading ? (
                    <LoadingSpinner size="sm" />
                  ) : (
                    <>
                      <Key className="h-4 w-4 mr-2" />
                      New Backup Codes
                    </>
                  )}
                </button>
                <button
                  onClick={() => setShowSetup(true)}
                  className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 transition-colors duration-200"
                >
                  Disable 2FA
                </button>
              </div>
            ) : (
              <button
                onClick={handleEnableTwoFactor}
                disabled={enableTwoFactorMutation.isLoading}
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-success hover:bg-green-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {enableTwoFactorMutation.isLoading ? (
                  <LoadingSpinner size="sm" />
                ) : (
                  <>
                    <Shield className="h-4 w-4 mr-2" />
                    Enable 2FA
                  </>
                )}
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Setup Modal */}
      {showSetup && setupData && !twoFactorStatus?.twoFactorEnabled && (
        <div className="medical-card">
          <h3 className="text-lg font-medium text-neutral-900 mb-4">
            Set Up Two-Factor Authentication
          </h3>
          
          <div className="space-y-6">
            {/* Step 1: QR Code */}
            <div>
              <h4 className="font-medium text-neutral-900 mb-2">
                Step 1: Scan QR Code
              </h4>
              <p className="text-sm text-neutral-600 mb-4">
                Scan this QR code with your authenticator app (Google Authenticator, Authy, etc.)
              </p>
              
              <div className="bg-white p-4 rounded-lg border text-center">
                <img 
                  src={`https://api.qrserver.com/v1/create-qr-code/?size=200x200&data=${encodeURIComponent(setupData.qrCodeUrl)}`}
                  alt="2FA QR Code"
                  className="mx-auto mb-4"
                />
                <div className="space-y-2">
                  <p className="text-xs text-neutral-500">
                    Can't scan? Enter this code manually:
                  </p>
                  <div className="flex items-center justify-center space-x-2">
                    <code className="bg-neutral-100 px-2 py-1 rounded text-xs font-mono">
                      {setupData.secretKey}
                    </code>
                    <button
                      onClick={() => copyToClipboard(setupData.secretKey)}
                      className="p-1 text-neutral-400 hover:text-primary-600"
                    >
                      <Copy className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* Step 2: Verification */}
            <div>
              <h4 className="font-medium text-neutral-900 mb-2">
                Step 2: Enter Verification Code
              </h4>
              <p className="text-sm text-neutral-600 mb-4">
                Enter the 6-digit code from your authenticator app
              </p>
              
              <form onSubmit={handleVerifySetup} className="space-y-4">
                <div>
                  <input
                    type="text"
                    value={verificationCode}
                    onChange={(e) => setVerificationCode(e.target.value.replace(/\D/g, '').slice(0, 6))}
                    placeholder="000000"
                    className="form-input text-center text-lg font-mono tracking-widest"
                    maxLength={6}
                    required
                  />
                </div>
                <div className="flex space-x-3">
                  <button
                    type="submit"
                    disabled={verificationCode.length !== 6 || verifySetupMutation.isLoading}
                    className="flex-1 inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {verifySetupMutation.isLoading ? (
                      <LoadingSpinner size="sm" />
                    ) : (
                      <>
                        <CheckCircle className="h-4 w-4 mr-2" />
                        Verify & Enable
                      </>
                    )}
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setShowSetup(false)
                      setSetupData(null)
                      setVerificationCode('')
                    }}
                    className="inline-flex items-center px-4 py-2 border border-neutral-300 text-sm font-medium rounded-md text-neutral-700 bg-white hover:bg-neutral-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 transition-colors duration-200"
                  >
                    Cancel
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>
      )}

      {/* Disable 2FA Modal */}
      {showSetup && twoFactorStatus?.twoFactorEnabled && (
        <div className="medical-card">
          <h3 className="text-lg font-medium text-neutral-900 mb-4">
            Disable Two-Factor Authentication
          </h3>
          
          <div className="bg-warning/10 border border-warning/20 rounded-lg p-4 mb-4">
            <p className="text-sm text-warning-dark">
              <strong>Warning:</strong> Disabling 2FA will make your account less secure. 
              You'll need to enter your password to confirm.
            </p>
          </div>

          <form onSubmit={handleDisableTwoFactor} className="space-y-4">
            <div>
              <label className="form-label">Current Password</label>
              <div className="relative">
                <input
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter your password"
                  className="form-input pr-10"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute inset-y-0 right-0 pr-3 flex items-center"
                >
                  {showPassword ? (
                    <EyeOff className="h-4 w-4 text-neutral-400" />
                  ) : (
                    <Eye className="h-4 w-4 text-neutral-400" />
                  )}
                </button>
              </div>
            </div>
            
            <div className="flex space-x-3">
              <button
                type="submit"
                disabled={disableTwoFactorMutation.isLoading}
                className="flex-1 inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {disableTwoFactorMutation.isLoading ? (
                  <LoadingSpinner size="sm" />
                ) : (
                  <>
                    <Shield className="h-4 w-4 mr-2" />
                    Disable 2FA
                  </>
                )}
              </button>
              <button
                type="button"
                onClick={() => {
                  setShowSetup(false)
                  setPassword('')
                }}
                className="inline-flex items-center px-4 py-2 border border-neutral-300 text-sm font-medium rounded-md text-neutral-700 bg-white hover:bg-neutral-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 transition-colors duration-200"
              >
                Cancel
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Backup Codes */}
      {showBackupCodes && setupData?.backupCodes && (
        <div className="medical-card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-medium text-neutral-900">
              Backup Codes
            </h3>
            <button
              onClick={copyBackupCodes}
              className="inline-flex items-center px-4 py-2 border border-neutral-300 text-sm font-medium rounded-md text-neutral-700 bg-white hover:bg-neutral-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 transition-colors duration-200"
            >
              <Copy className="h-4 w-4 mr-2" />
              Copy All
            </button>
          </div>
          
          <div className="bg-neutral-50 rounded-lg p-4 mb-4">
            <div className="flex items-start space-x-2 mb-3">
              <Key className="h-5 w-5 text-warning flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-sm text-neutral-900 font-medium">
                  Important: Save these backup codes
                </p>
                <p className="text-xs text-neutral-600">
                  These codes can be used to access your account if you lose your authenticator device. 
                  Each code can only be used once.
                </p>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-2">
              {setupData.backupCodes.map((code, index) => (
                <div key={index} className="flex items-center space-x-2">
                  <code className="bg-white px-3 py-2 rounded border text-sm font-mono flex-1">
                    {code}
                  </code>
                  <button
                    onClick={() => copyToClipboard(code)}
                    className="p-1 text-neutral-400 hover:text-primary-600"
                  >
                    <Copy className="h-3 w-3" />
                  </button>
                </div>
              ))}
            </div>
          </div>
          
          <div className="flex justify-end">
            <button
              onClick={() => setShowBackupCodes(false)}
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-success hover:bg-green-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 transition-colors duration-200"
            >
              <CheckCircle className="h-4 w-4 mr-2" />
              I've Saved These Codes
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

export default TwoFactorSetup