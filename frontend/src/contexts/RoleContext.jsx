import React, { createContext, useContext, useState } from 'react';

const RoleContext = createContext();

export const RoleProvider = ({ children }) => {
  const [userRole, setUserRole] = useState('patient'); // 'patient' or 'physician'

  const updateRole = (role) => {
    if (role === 'patient' || role === 'physician') {
      setUserRole(role);
    }
  };

  return (
    <RoleContext.Provider value={{ userRole, updateRole }}>
      {children}
    </RoleContext.Provider>
  );
};

export const useRole = () => {
  const context = useContext(RoleContext);
  if (!context) {
    throw new Error('useRole must be used within a RoleProvider');
  }
  return context;
};

export default RoleProvider;
