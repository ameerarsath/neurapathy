function Avatar({ size = 'md', name = 'User', src }) {
 
  const getInitials = (name) => {
    return name
      .split(' ')
      .map(part => part[0])
      .join('')
      .toUpperCase()
      .substring(0, 2);
  }
  const sizeClasses = {
    sm: 'w-8 h-8 text-xs',
    md: 'w-10 h-10 text-sm',
    lg: 'w-12 h-12 text-base',
  }

  return (
    <div className={`relative inline-flex items-center justify-center bg-primary-100 text-primary-700 font-medium rounded-full ${sizeClasses[size]}`}>
      {src ? (
        <img 
          src={src} 
          alt={name} 
          className="w-full h-full object-cover rounded-full"
        />
      ) : (
        <span>{getInitials(name)}</span>
      )}
    </div>
  )
}
export default Avatar