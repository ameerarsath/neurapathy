function Button({ 
  children, 
  variant = 'primary', 
  size = 'md', 
  className = '', 
  leftIcon,
  rightIcon,
  isLoading = false,
  disabled = false,
  onClick,
  type = 'button',
  ...props 
}) {
  const baseClasses = 'btn'
  
  const variantClasses = {
    primary: 'btn-primary',
    secondary: 'btn-secondary',
    outline: 'btn-outline',
    ghost: 'text-primary-500 hover:bg-primary-50 bg-transparent',
    danger: 'bg-error-500 hover:bg-error-600 text-white',
  }
  
  const sizeClasses = {
    sm: 'px-2 py-1 text-sm',
    md: 'px-4 py-2',
    lg: 'px-6 py-3 text-lg',
  }
  
  const classes = `
    ${baseClasses} 
    ${variantClasses[variant] || variantClasses.primary}
    ${sizeClasses[size] || sizeClasses.md}
    ${disabled || isLoading ? 'opacity-60 cursor-not-allowed' : ''}
    ${className}
  `

  return (
    <button 
      className={classes}
      onClick={onClick}
      disabled={disabled || isLoading}
      type={type}
      {...props}
    >
      
      {leftIcon && <span className="mr-2">{leftIcon}</span>}
      
      {/* Loading Spinner */}
      {isLoading ? (
        <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-current" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
      ) : null}
      
      {/* Button Text */}
      {children}
      
      {/* Right Icon */}
      {rightIcon && <span className="ml-2">{rightIcon}</span>}
    </button>
  )
}

export default Button