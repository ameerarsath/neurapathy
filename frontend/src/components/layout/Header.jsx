import { useState } from 'react'
import Avatar from '../common/Avatar'

function Header({ toggleSidebar }) {
  const [showDropdown, setShowDropdown] = useState(false)
  
  return (
    <header className="bg-white border-b border-neutral-200 px-4 py-3 flex items-center justify-between shadow-sm">
      <div className="flex items-center">
        <button 
          className="block md:hidden mr-2 text-neutral-600 hover:text-neutral-900"
          onClick={toggleSidebar}
          aria-label="Toggle sidebar"
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>
        <h1 className="text-lg font-semibold text-primary-600">NeuropathyCare</h1>
      </div>

      <div className="flex items-center">
        <div className="relative">
          <button 
            className="flex items-center"
            onClick={() => setShowDropdown(!showDropdown)}
          >
            <span className="mr-2 hidden md:block text-sm text-neutral-700">Dr. Smith</span>
            <Avatar size="sm" name="Dr. Smith" />
          </button>

          {showDropdown && (
            <div className="absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg py-1 z-10 border border-neutral-200 animate-fade-in">
              <a 
                href="#profile" 
                className="block px-4 py-2 text-sm text-neutral-700 hover:bg-neutral-100"
              >
                Your Profile
              </a>
              <a 
                href="#settings" 
                className="block px-4 py-2 text-sm text-neutral-700 hover:bg-neutral-100"
              >
                Settings
              </a>
              <div className="border-t border-neutral-200 my-1"></div>
              <a 
                href="#logout" 
                className="block px-4 py-2 text-sm text-neutral-700 hover:bg-neutral-100"
              >
                Sign out
              </a>
            </div>
          )}
        </div>
      </div>
    </header>
  )
}

export default Header