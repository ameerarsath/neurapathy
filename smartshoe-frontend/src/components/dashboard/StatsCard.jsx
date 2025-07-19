import React from 'react'
import { TrendingUp, TrendingDown } from 'lucide-react'
import clsx from 'clsx'

const StatsCard = ({ 
  title, 
  value, 
  icon: Icon, 
  color = 'primary', 
  trend, 
  subtitle,
  className = '' 
}) => {
  const colorClasses = {
    primary: 'text-primary-600 bg-primary-50',
    secondary: 'text-secondary-600 bg-secondary-50',
    success: 'text-green-600 bg-green-50',
    warning: 'text-yellow-600 bg-yellow-50',
    error: 'text-red-600 bg-red-50'
  }

  const trendColor = trend >= 0 ? 'text-green-600' : 'text-red-600'
  const TrendIcon = trend >= 0 ? TrendingUp : TrendingDown

  return (
    <div className={clsx('medical-card', className)}>
      <div className="flex items-center">
        <div className={clsx('p-3 rounded-lg', colorClasses[color])}>
          <Icon className="h-6 w-6" />
        </div>
        <div className="ml-4 flex-1">
          <p className="text-sm font-medium text-neutral-600">{title}</p>
          <p className="text-2xl font-bold text-neutral-900">{value}</p>
          {subtitle && (
            <p className="text-xs text-neutral-500 mt-1">{subtitle}</p>
          )}
        </div>
      </div>
      
      {trend !== undefined && (
        <div className="mt-4 flex items-center">
          <div className={clsx('flex items-center text-sm', trendColor)}>
            <TrendIcon className="h-4 w-4 mr-1" />
            <span>{Math.abs(trend)}%</span>
          </div>
          <span className="text-xs text-neutral-500 ml-2">
            vs last period
          </span>
        </div>
      )}
    </div>
  )
}

export default StatsCard