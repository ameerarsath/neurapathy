import { format, formatDistanceToNow, isValid, parseISO } from 'date-fns'

/**
 * Utility functions for date formatting and manipulation
 */

export const formatDate = (date, formatString = 'MMM dd, yyyy') => {
  if (!date) return ''
  
  const parsedDate = typeof date === 'string' ? parseISO(date) : date
  
  if (!isValid(parsedDate)) return ''
  
  return format(parsedDate, formatString)
}

export const formatTime = (date, formatString = 'HH:mm') => {
  if (!date) return ''
  
  const parsedDate = typeof date === 'string' ? parseISO(date) : date
  
  if (!isValid(parsedDate)) return ''
  
  return format(parsedDate, formatString)
}

export const formatDateTime = (date, formatString = 'MMM dd, yyyy HH:mm') => {
  if (!date) return ''
  
  const parsedDate = typeof date === 'string' ? parseISO(date) : date
  
  if (!isValid(parsedDate)) return ''
  
  return format(parsedDate, formatString)
}

export const formatRelativeTime = (date) => {
  if (!date) return ''
  
  const parsedDate = typeof date === 'string' ? parseISO(date) : date
  
  if (!isValid(parsedDate)) return ''
  
  return formatDistanceToNow(parsedDate, { addSuffix: true })
}

export const formatMedicalDate = (date) => {
  return formatDate(date, 'yyyy-MM-dd')
}

export const formatTestSessionDate = (date) => {
  return formatDateTime(date, 'MMM dd, yyyy \'at\' HH:mm')
}

export const getTimeAgo = (timestamp) => {
  const now = new Date()
  const time = new Date(timestamp)
  const diffInMinutes = Math.floor((now - time) / (1000 * 60))

  if (diffInMinutes < 1) return 'Just now'
  if (diffInMinutes < 60) return `${diffInMinutes}m ago`
  if (diffInMinutes < 1440) return `${Math.floor(diffInMinutes / 60)}h ago`
  return `${Math.floor(diffInMinutes / 1440)}d ago`
}

export const isToday = (date) => {
  const today = new Date()
  const checkDate = typeof date === 'string' ? parseISO(date) : date
  
  return today.toDateString() === checkDate.toDateString()
}

export const isThisWeek = (date) => {
  const now = new Date()
  const checkDate = typeof date === 'string' ? parseISO(date) : date
  const daysDifference = Math.floor((now - checkDate) / (1000 * 60 * 60 * 24))
  
  return daysDifference <= 7
}

export const getDateRange = (range) => {
  const now = new Date()
  const start = new Date()
  
  switch (range) {
    case 'day':
      start.setHours(0, 0, 0, 0)
      break
    case 'week':
      start.setDate(now.getDate() - 7)
      start.setHours(0, 0, 0, 0)
      break
    case 'month':
      start.setMonth(now.getMonth() - 1)
      start.setHours(0, 0, 0, 0)
      break
    case 'year':
      start.setFullYear(now.getFullYear() - 1)
      start.setHours(0, 0, 0, 0)
      break
    default:
      start.setHours(0, 0, 0, 0)
  }
  
  return { start, end: now }
}