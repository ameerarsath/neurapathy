import { forwardRef } from 'react'

const Input = forwardRef(({
  label,
  type = 'text',
  id,
  name,
  placeholder,
  error,
  helperText,
  className = '',
  required = false,
  disabled = false,
  ...props
}, ref) => {
  return (
    <div className={`mb-4 ${className}`}>
    
      {label && (
        <label 
          htmlFor={id} 
          className="block text-sm font-medium text-neutral-700 mb-1"
        >
          {label}
          {required && <span className="text-error-500 ml-1">*</span>}
        </label>
      )}
      
    
      <input
        ref={ref}
        type={type}
        id={id}
        name={name}
        placeholder={placeholder}
        disabled={disabled}
        className={`
          input
          ${error ? 'border-error-500 focus:ring-error-500' : 'focus:ring-primary-500'}
          ${disabled ? 'bg-neutral-100 cursor-not-allowed' : ''}
        `}
        {...props}
      />
      {(error || helperText) && (
        <p className={`mt-1 text-sm ${error ? 'text-error-500' : 'text-neutral-500'}`}>
          {error || helperText}
        </p>
      )}
    </div>
  )
})

export default Input