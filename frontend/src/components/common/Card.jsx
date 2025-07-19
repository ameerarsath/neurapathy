function Card({ 
  children, 
  title,
  subtitle,
  footer,
  className = '',
  isHoverable = false,
  ...props 
}) {
  return (
    <div 
      className={`
        bg-white rounded-lg border border-neutral-200 overflow-hidden
        ${isHoverable ? 'transform transition-all duration-200 hover:-translate-y-1 hover:shadow-lg' : 'shadow-card'}
        ${className}
      `}
      {...props}
    >
      {(title || subtitle) && (
        <div className="px-4 py-3 border-b border-neutral-200">
          {title && <h3 className="font-medium text-neutral-900">{title}</h3>}
          {subtitle && <p className="text-sm text-neutral-500 mt-1">{subtitle}</p>}
        </div>
      )}
      
      
      <div className="p-4">
        {children}
      </div>
      
    
      {footer && (
        <div className="px-4 py-3 bg-neutral-50 border-t border-neutral-200">
          {footer}
        </div>
      )}
    </div>
  )
}

export default Card