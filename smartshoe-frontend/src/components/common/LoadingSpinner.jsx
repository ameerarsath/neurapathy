import React from 'react'
import { Loader2 } from 'lucide-react'
import clsx from 'clsx'

const LoadingSpinner = ({ 
  size = 'md', 
  color = 'primary', 
  className = '',
  ...props 
}) => {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
    xl: 'w-12 h-12'
  }

  const colorClasses = {
    primary: 'text-primary-500',
    secondary: 'text-secondary-500',
    success: 'text-success',
    warning: 'text-warning',
    error: 'text-error',
    neutral: 'text-neutral-500'
  }

  return (
    <Loader2 
      className={clsx(
        'animate-spin',
        sizeClasses[size],
        colorClasses[color],
        className
      )}
      {...props}
    />
  )
}

export default LoadingSpinner