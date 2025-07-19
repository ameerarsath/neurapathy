package com.smartshoe.api.service;

import com.smartshoe.api.entity.User;
import com.smartshoe.api.repository.UserRepository;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Custom UserDetailsService for Spring Security integration
 */
@Service
public class CustomUserDetailsService implements UserDetailsService {

    private final UserRepository userRepository;

    public CustomUserDetailsService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    @Override
    public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new UsernameNotFoundException("User not found: " + username));

        return new CustomUserPrincipal(user);
    }

    /**
     * Custom UserDetails implementation
     */
    public static class CustomUserPrincipal implements UserDetails {
        private final User user;

        public CustomUserPrincipal(User user) {
            this.user = user;
        }

        @Override
        public Collection<? extends GrantedAuthority> getAuthorities() {
            List<GrantedAuthority> authorities = new ArrayList<>();
            
            if (user.getRoles() != null) {
                for (User.Role role : user.getRoles()) {
                    authorities.add(new SimpleGrantedAuthority("ROLE_" + role.name()));
                }
            }
            
            return authorities;
        }

        @Override
        public String getPassword() {
            return user.getPassword();
        }

        @Override
        public String getUsername() {
            return user.getUsername();
        }

        @Override
        public boolean isAccountNonExpired() {
            return user.getAccountNonExpired();
        }

        @Override
        public boolean isAccountNonLocked() {
            return user.getAccountNonLocked() && !user.isAccountLocked();
        }

        @Override
        public boolean isCredentialsNonExpired() {
            return user.getCredentialsNonExpired() && !user.isPasswordExpired();
        }

        @Override
        public boolean isEnabled() {
            return user.getEnabled();
        }

        // Getter for the User entity
        public User getUser() {
            return user;
        }
    }
}