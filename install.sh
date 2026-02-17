#!/bin/bash
# ==========================================
# ZYNARA MEGA AI — RUNPOD AUTO INSTALL SCRIPT v2.0
# Enhanced with Security, Monitoring & Performance
# (Assumes: already inside /workspace/Zynara)
# ==========================================

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO:${NC} $1"
}

# Banner
show_banner() {
    echo -e "${PURPLE}"
    cat <<'EOF'
╔══════════════════════════════════════════════════════════════╗
║                ZYNARA MEGA AI — ENHANCED INSTALLER           ║
║                      Version 2.0 — Production Ready            ║
╠══════════════════════════════════════════════════════════════╣
║  Features:                                                    ║
║    • Security Hardening                                       ║
║    • Performance Optimization                                ║
║    • Monitoring & Alerting                                    ║
║    • Auto-scaling Support                                     ║
║    • SSL/TLS Configuration                                    ║
║    • Backup & Recovery                                        ║
║    • Health Checks                                            ║
║    • Multi-environment Support                                ║
╚══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

# Configuration
CONFIG_FILE="install.conf"
ENV_FILE=".env"
BACKUP_DIR="backups"
LOG_DIR="logs"
TEMP_DIR="/tmp/zynara_install"

# Default configuration
DEFAULT_CONFIG=$(cat <<'EOF'
# Zynara Mega AI Configuration
ENVIRONMENT=production
PORT=7860
WORKERS=4
WORKER_CLASS=uvicorn.workers.UvicornWorker
MAX_REQUESTS=1000
MAX_REQUESTS_JITTER=100
TIMEOUT=300
KEEP_ALIVE=2

# Security
ENABLE_SSL=false
SSL_CERT_PATH=/etc/ssl/certs/zynara.crt
SSL_KEY_PATH=/etc/ssl/private/zynara.key
ENABLE_RATE_LIMITING=true
ENABLE_CORS=true

# Performance
ENABLE_REDIS=true
REDIS_PORT=6379
REDIS_MAX_MEMORY=256mb
ENABLE_COMPRESSION=true
COMPRESSION_LEVEL=6

# Monitoring
ENABLE_PROMETHEUS=true
PROMETHEUS_PORT=9090
ENABLE_GRAFANA=true
GRAFANA_PORT=3000
ENABLE_HEALTH_CHECKS=true

# Backup
ENABLE_AUTO_BACKUP=true
BACKUP_INTERVAL=24
BACKUP_RETENTION=7

# Logging
LOG_LEVEL=info
LOG_FORMAT=json
ENABLE_ACCESS_LOG=true
MAX_LOG_SIZE=100m
LOG_RETENTION=30
EOF
)

# Parse command line arguments
parse_args() {
    ENVIRONMENT="production"
    ENABLE_SSL=false
    SKIP_DEPS=false
    FORCE_REINSTALL=false
    BACKUP_BEFORE_INSTALL=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            --enable-ssl)
                ENABLE_SSL=true
                shift
                ;;
            --skip-deps)
                SKIP_DEPS=true
                shift
                ;;
            --force)
                FORCE_REINSTALL=true
                shift
                ;;
            --backup)
                BACKUP_BEFORE_INSTALL=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                ;;
        esac
    done
}

# Show help
show_help() {
    cat <<'EOF'
Zynara Mega AI Installer v2.0

USAGE:
    ./install.sh [OPTIONS]

OPTIONS:
    --environment ENV    Set environment (development|staging|production) [default: production]
    --enable-ssl         Enable SSL/TLS configuration
    --skip-deps          Skip system dependencies installation
    --force              Force reinstallation
    --backup             Create backup before installation
    --help, -h           Show this help message

EXAMPLES:
    ./install.sh                           # Standard installation
    ./install.sh --environment staging     # Staging environment
    ./install.sh --enable-ssl --backup     # SSL enabled with backup
    ./install.sh --skip-deps --force       # Quick reinstall

EOF
}

# Check system requirements
check_requirements() {
    log "Checking system requirements..."
    
    # Check OS
    if ! command -v apt &> /dev/null; then
        error "This installer requires apt package manager (Ubuntu/Debian)"
    fi
    
    # Check memory
    MEMORY=$(free -m | awk 'NR==2{printf "%.0f", $2}')
    if [ "$MEMORY" -lt 2048 ]; then
        warn "System has less than 2GB RAM. Performance may be affected."
    fi
    
    # Check disk space
    DISK=$(df / | awk 'NR==2{print $4}')
    if [ "$DISK" -lt 5242880 ]; then  # 5GB in KB
        error "Insufficient disk space. Requires at least 5GB free."
    fi
    
    # Check if running as root
    if [ "$EUID" -eq 0 ]; then
        warn "Running as root is not recommended. Consider using a non-root user."
    fi
    
    log "System requirements check passed"
}

# Create backup
create_backup() {
    if [ "$BACKUP_BEFORE_INSTALL" = true ] || [ -d "venv" ]; then
        log "Creating backup..."
        BACKUP_NAME="backup_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$BACKUP_DIR"
        
        # Backup important files
        tar -czf "$BACKUP_DIR/$BACKUP_NAME.tar.gz" \
            venv \
            .env \
            logs \
            --exclude='*.pyc' \
            --exclude='__pycache__' \
            2>/dev/null || true
        
        log "Backup created: $BACKUP_DIR/$BACKUP_NAME.tar.gz"
    fi
}

# Update system
update_system() {
    if [ "$SKIP_DEPS" = false ]; then
        log "Updating system packages..."
        apt update && apt upgrade -y
        apt autoremove -y
        apt autoclean
    fi
}

# Install system dependencies
install_system_deps() {
    if [ "$SKIP_DEPS" = false ]; then
        log "Installing system dependencies..."
        
        # Core dependencies
        apt install -y \
            python3 \
            python3-pip \
            python3-venv \
            python3-dev \
            git \
            curl \
            wget \
            unzip \
            ca-certificates \
            build-essential \
            software-properties-common \
            apt-transport-https \
            gnupg \
            lsb-release
            
        # Media dependencies
        apt install -y \
            ffmpeg \
            libsm6 \
            libxext6 \
            libxrender-dev \
            libglib2.0-0 \
            libsndfile1
            
        # Security dependencies
        apt install -y \
            certbot \
            nginx \
            fail2ban \
            ufw
            
        # Monitoring dependencies
        apt install -y \
            htop \
            iotop \
            nethogs \
            sysstat
    fi
}

# Setup firewall
setup_firewall() {
    log "Configuring firewall..."
    
    # Reset UFW
    ufw --force reset
    
    # Default policies
    ufw default deny incoming
    ufw default allow outgoing
    
    # Allow SSH
    ufw allow ssh
    
    # Allow application ports
    ufw allow "$PORT/tcp"
    
    if [ "$ENABLE_SSL" = true ]; then
        ufw allow 443/tcp
    fi
    
    if [ "${ENABLE_PROMETHEUS:-true}" = true ]; then
        ufw allow 9090/tcp
    fi
    
    if [ "${ENABLE_GRAFANA:-true}" = true ]; then
        ufw allow 3000/tcp
    fi
    
    # Enable firewall
    ufw --force enable
    
    log "Firewall configured"
}

# Create configuration
create_config() {
    log "Creating configuration files..."
    
    # Create install config
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "$DEFAULT_CONFIG" > "$CONFIG_FILE"
    fi
    
    # Source configuration
    source "$CONFIG_FILE"
    
    # Create environment file
    if [ ! -f "$ENV_FILE" ] || [ "$FORCE_REINSTALL" = true ]; then
        log "Creating .env file..."
        
        cat > "$ENV_FILE" <<EOF
# Zynara Mega AI Environment Configuration
APP_NAME=Zynara Mega AI
APP_AUTHOR=GoldYLocks
APP_DESCRIPTION=Multi-modal AI backend
APP_VERSION=2.0.0

# Server Configuration
ENVIRONMENT=$ENVIRONMENT
PORT=$PORT
WORKERS=$WORKERS
WORKER_CLASS=$WORKER_CLASS
MAX_REQUESTS=$MAX_REQUESTS
MAX_REQUESTS_JITTER=$MAX_REQUESTS_JITTER
TIMEOUT=$TIMEOUT
KEEP_ALIVE=$KEEP_ALIVE

# Security
SECRET_KEY=$(openssl rand -hex 32)
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
ENABLE_SSL=$ENABLE_SSL
ENABLE_RATE_LIMITING=$ENABLE_RATE_LIMITING
ENABLE_CORS=$ENABLE_CORS

# Database
DATABASE_URL=postgresql://zynara:$(openssl rand -base64 32)@localhost:5432/zynara_db
REDIS_URL=redis://localhost:$REDIS_PORT/0

# API Keys (Please replace with actual keys)
OPENAI_API_KEY=sk-XXXXXXXXXXXXXXXXXXXX
GROQ_API_KEY=gsk_XXXXXXXXXXXXXXXXXXXX
SUPABASE_URL=https://your-supabase-url.supabase.co
SUPABASE_KEY=your-supabase-key
ELEVEN_API_KEY=XXXXXXXXXXXXXXXXXXXX

# Performance
ENABLE_REDIS=$ENABLE_REDIS
REDIS_MAX_MEMORY=$REDIS_MAX_MEMORY
ENABLE_COMPRESSION=$ENABLE_COMPRESSION
COMPRESSION_LEVEL=$COMPRESSION_LEVEL

# Monitoring
ENABLE_PROMETHEUS=$ENABLE_PROMETHEUS
PROMETHEUS_PORT=$PROMETHEUS_PORT
ENABLE_GRAFANA=$ENABLE_GRAFANA
GRAFANA_PORT=$GRAFANA_PORT
ENABLE_HEALTH_CHECKS=$ENABLE_HEALTH_CHECKS

# Logging
LOG_LEVEL=$LOG_LEVEL
LOG_FORMAT=$LOG_FORMAT
ENABLE_ACCESS_LOG=$ENABLE_ACCESS_LOG
LOG_DIR=$LOG_DIR

# Storage
IMAGES_DIR=/tmp/generated_images
TEMP_DIR=/tmp/zynara
BACKUP_DIR=$BACKUP_DIR

# CORS
ALLOWED_ORIGINS=*
ALLOWED_METHODS=GET,POST,PUT,DELETE,OPTIONS
ALLOWED_HEADERS=*

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_BURST=10

# SSL/TLS
SSL_CERT_PATH=$SSL_CERT_PATH
SSL_KEY_PATH=$SSL_KEY_PATH

# Backup
ENABLE_AUTO_BACKUP=$ENABLE_AUTO_BACKUP
BACKUP_INTERVAL=$BACKUP_INTERVAL
BACKUP_RETENTION=$BACKUP_RETENTION
EOF
        
        log "Environment file created"
    fi
}

# Setup Python environment
setup_python_env() {
    log "Setting up Python environment..."
    
    # Remove existing venv if forcing reinstall
    if [ "$FORCE_REINSTALL" = true ] && [ -d "venv" ]; then
        rm -rf venv
    fi
    
    # Create virtual environment
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip and tools
    pip install --upgrade pip setuptools wheel pip-tools
    
    # Install Python dependencies
    if [ -f "requirements.txt" ]; then
        log "Installing Python requirements..."
        pip install -r requirements.txt
    fi
    
    # Install additional production dependencies
    pip install \
        gunicorn \
        uvicorn[standard] \
        prometheus-client \
        psutil \
        pydantic-settings \
        python-multipart \
        orjson \
        aiofiles \
        redis \
        asyncpg
    
    log "Python environment setup complete"
}

# Setup Redis
setup_redis() {
    if [ "${ENABLE_REDIS:-true}" = true ]; then
        log "Setting up Redis..."
        
        # Install Redis if not present
        if ! command -v redis-server &> /dev/null; then
            apt install -y redis-server
        fi
        
        # Configure Redis
        cat > /etc/redis/redis.conf <<EOF
# Redis configuration for Zynara Mega AI
port $REDIS_PORT
bind 127.0.0.1
timeout 300
keepalive 60
maxmemory $REDIS_MAX_MEMORY
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
EOF
        
        # Enable and start Redis
        systemctl enable redis-server
        systemctl restart redis-server
        
        # Test Redis connection
        if redis-cli ping | grep -q "PONG"; then
            log "Redis is running successfully"
        else
            error "Redis failed to start"
        fi
    fi
}

# Setup SSL/TLS
setup_ssl() {
    if [ "$ENABLE_SSL" = true ]; then
        log "Setting up SSL/TLS..."
        
        # Generate self-signed certificate for development
        if [ "$ENVIRONMENT" != "production" ]; then
            mkdir -p /etc/ssl/certs /etc/ssl/private
            
            openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
                -keyout "$SSL_KEY_PATH" \
                -out "$SSL_CERT_PATH" \
                -subj "/C=US/ST=State/L=City/O=Zynara/CN=localhost"
            
            log "Self-signed SSL certificate generated"
        else
            # For production, use Let's Encrypt
            if command -v certbot &> /dev/null; then
                log "Please configure SSL certificate for production domain"
                info "Run: certbot --nginx -d yourdomain.com"
            fi
        fi
    fi
}

# Setup Nginx
setup_nginx() {
    log "Setting up Nginx..."
    
    # Create Nginx configuration
    cat > /etc/nginx/sites-available/zynara <<EOF
server {
    listen 80;
    server_name _;
    
    # Redirect to HTTPS if SSL is enabled
    if ($ENABLE_SSL) {
        return 301 https://\$host\$request_uri;
    }
    
    # Proxy to application
    location / {
        proxy_pass http://127.0.0.1:$PORT;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # Enable compression
        gzip on;
        gzip_types text/plain text/css application/json application/javascript;
        gzip_comp_level $COMPRESSION_LEVEL;
    }
    
    # Static files
    location /static/ {
        alias /workspace/Zynara/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}

# HTTPS server (if SSL is enabled)
server {
    listen 443 ssl http2;
    server_name _;
    
    ssl_certificate $SSL_CERT_PATH;
    ssl_certificate_key $SSL_KEY_PATH;
    
    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
    
    # Proxy configuration (same as HTTP)
    location / {
        proxy_pass http://127.0.0.1:$PORT;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
    }
}
EOF
    
    # Enable site
    ln -sf /etc/nginx/sites-available/zynara /etc/nginx/sites-enabled/
    rm -f /etc/nginx/sites-enabled/default
    
    # Test and restart Nginx
    nginx -t && systemctl restart nginx
    
    log "Nginx configured successfully"
}

# Setup monitoring
setup_monitoring() {
    log "Setting up monitoring..."
    
    # Create monitoring directory
    mkdir -p monitoring
    
    # Create Prometheus configuration
    if [ "${ENABLE_PROMETHEUS:-true}" = true ]; then
        cat > monitoring/prometheus.yml <<EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'zynara-app'
    static_configs:
      - targets: ['localhost:$PORT']
    metrics_path: '/metrics'
    scrape_interval: 5s
    
  - job_name: 'redis'
    static_configs:
      - targets: ['localhost:$REDIS_PORT']
    scrape_interval: 10s
    
  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']
EOF
        
        log "Prometheus configuration created"
    fi
    
    # Create Grafana dashboard configuration
    if [ "${ENABLE_GRAFANA:-true}" = true ]; then
        cat > monitoring/grafana-dashboard.json <<'EOF'
{
  "dashboard": {
    "title": "Zynara Mega AI Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      }
    ]
  }
}
EOF
        
        log "Grafana dashboard configuration created"
    fi
}

# Create directories
create_directories() {
    log "Creating runtime directories..."
    
    mkdir -p \
        "$IMAGES_DIR" \
        "$TEMP_DIR" \
        "$LOG_DIR" \
        "$BACKUP_DIR" \
        static \
        monitoring \
        scripts
    
    # Set permissions
    chmod 755 "$IMAGES_DIR" "$TEMP_DIR" "$LOG_DIR"
    chmod 700 "$BACKUP_DIR"
    
    log "Directories created"
}

# Setup log rotation
setup_log_rotation() {
    log "Setting up log rotation..."
    
    cat > /etc/logrotate.d/zynara <<EOF
 $LOG_DIR/*.log {
    daily
    missingok
    rotate $LOG_RETENTION
    compress
    delaycompress
    notifempty
    create 644 $(whoami) $(whoami)
    postrotate
        systemctl reload zynara || true
    endscript
}
EOF
    
    log "Log rotation configured"
}

# Create systemd service
create_systemd_service() {
    log "Creating systemd service..."
    
    cat > /etc/systemd/system/zynara.service <<EOF
[Unit]
Description=Zynara Mega AI Backend
After=network.target redis.service nginx.service
Wants=redis.service nginx.service

[Service]
Type=exec
User=$(whoami)
Group=$(whoami)
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin
EnvironmentFile=$(pwd)/.env
ExecStart=$(pwd)/venv/bin/gunicorn app:app \\
    --bind 127.0.0.1:$PORT \\
    --workers $WORKERS \\
    --worker-class $WORKER_CLASS \\
    --worker-connections 1000 \\
    --max-requests $MAX_REQUESTS \\
    --max-requests-jitter $MAX_REQUESTS_JITTER \\
    --timeout $TIMEOUT \\
    --keep-alive $KEEP_ALIVE \\
    --access-logfile $LOG_DIR/access.log \\
    --error-logfile $LOG_DIR/error.log \\
    --log-level $LOG_LEVEL \\
    --capture-output
ExecReload=/bin/kill -s HUP \$MAINPID
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=zynara

[Install]
WantedBy=multi-user.target
EOF
    
    # Reload systemd and enable service
    systemctl daemon-reload
    systemctl enable zynara
    
    log "Systemd service created"
}

# Setup backup scripts
setup_backup_scripts() {
    if [ "${ENABLE_AUTO_BACKUP:-true}" = true ]; then
        log "Setting up backup scripts..."
        
        cat > scripts/backup.sh <<'EOF'
#!/bin/bash
# Zynara Mega AI Backup Script

BACKUP_DIR="/workspace/Zynara/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="zynara_backup_$DATE"

# Create backup
mkdir -p "$BACKUP_DIR"

# Backup database
if command -v pg_dump &> /dev/null; then
    pg_dump zynara_db > "$BACKUP_DIR/${BACKUP_NAME}_db.sql"
fi

# Backup files
tar -czf "$BACKUP_DIR/${BACKUP_NAME}_files.tar.gz" \
    .env \
    logs \
    static \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='venv' \
    --exclude='backups'

# Clean old backups
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +7 -delete
find "$BACKUP_DIR" -name "*.sql" -mtime +7 -delete

echo "Backup completed: $BACKUP_NAME"
EOF
        
        chmod +x scripts/backup.sh
        
        # Create cron job
        (crontab -l 2>/dev/null; echo "0 2 * * * $(pwd)/scripts/backup.sh") | crontab -
        
        log "Backup scripts configured"
    fi
}

# Health check function
health_check() {
    log "Performing health check..."
    
    # Check if service is running
    if systemctl is-active --quiet zynara; then
        log "✅ Zynara service is running"
    else
        error "❌ Zynara service is not running"
    fi
    
    # Check API endpoint
    if curl -f "http://localhost:$PORT/health" &> /dev/null; then
        log "✅ API health check passed"
    else
        warn "⚠️ API health check failed"
    fi
    
    # Check Redis
    if [ "${ENABLE_REDIS:-true}" = true ]; then
        if redis-cli ping | grep -q "PONG"; then
            log "✅ Redis is responding"
        else
            warn "⚠️ Redis is not responding"
        fi
    fi
    
    # Check Nginx
    if systemctl is-active --quiet nginx; then
        log "✅ Nginx is running"
    else
        warn "⚠️ Nginx is not running"
    fi
}

# Start services
start_services() {
    log "Starting Zynara Mega AI services..."
    
    # Start application
    systemctl start zynara
    
    # Wait for service to start
    sleep 5
    
    # Check if started successfully
    if systemctl is-active --quiet zynara; then
        log "✅ Zynara Mega AI started successfully"
    else
        error "❌ Failed to start Zynara Mega AI"
    fi
}

# Show final information
show_final_info() {
    # Get public IP
    PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "localhost")
    
    # Get status
    SERVICE_STATUS=$(systemctl is-active zynara)
    NGINX_STATUS=$(systemctl is-active nginx)
    REDIS_STATUS=$(systemctl is-active redis-server 2>/dev/null || echo "N/A")
    
    echo -e "${CYAN}"
    cat <<'EOF'
╔══════════════════════════════════════════════════════════════╗
║                     ✅ INSTALLATION COMPLETE                  ║
╠══════════════════════════════════════════════════════════════╣
EOF
    echo -e "${NC}"
    
    echo -e "${GREEN}🌍 Application URLs:${NC}"
    echo "   • Main API: http://$PUBLIC_IP:$PORT"
    if [ "$ENABLE_SSL" = true ]; then
        echo "   • HTTPS: https://$PUBLIC_IP"
    fi
    echo "   • API Docs: http://$PUBLIC_IP:$PORT/docs"
    echo "   • Health: http://$PUBLIC_IP:$PORT/health"
    
    if [ "${ENABLE_PROMETHEUS:-true}" = true ]; then
        echo "   • Prometheus: http://$PUBLIC_IP:9090"
    fi
    
    if [ "${ENABLE_GRAFANA:-true}" = true ]; then
        echo "   • Grafana: http://$PUBLIC_IP:3000"
    fi
    
    echo -e "\n${GREEN}📊 Service Status:${NC}"
    echo "   • Zynara Service: $SERVICE_STATUS"
    echo "   • Nginx: $NGINX_STATUS"
    echo "   • Redis: $REDIS_STATUS"
    
    echo -e "\n${GREEN}🔧 Management Commands:${NC}"
    echo "   • View logs: journalctl -u zynara -f"
    echo "   • Restart: systemctl restart zynara"
    echo "   • Status: systemctl status zynara"
    echo "   • Config: nano .env"
    echo "   • Backup: ./scripts/backup.sh"
    
    echo -e "\n${GREEN}📁 Important Paths:${NC}"
    echo "   • Application: $(pwd)"
    echo "   • Logs: $LOG_DIR"
    echo "   • Backups: $BACKUP_DIR"
    echo "   • Config: $ENV_FILE"
    echo "   • Nginx Config: /etc/nginx/sites-available/zynara"
    
    echo -e "\n${YELLOW}⚠️ Security Notes:${NC}"
    echo "   • Change default API keys in .env"
    echo "   • Configure SSL certificate for production"
    echo "   • Review firewall rules: ufw status"
    echo "   • Monitor logs for suspicious activity"
    
    echo -e "\n${PURPLE}🎉 Zynara Mega AI is ready for production!${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
}

# Cleanup function
cleanup() {
    log "Cleaning up temporary files..."
    rm -rf "$TEMP_DIR"
}

# Main installation function
main() {
    # Set up error handling
    trap cleanup EXIT
    
    # Show banner
    show_banner
    
    # Parse arguments
    parse_args "$@"
    
    # Installation steps
    check_requirements
    create_backup
    update_system
    install_system_deps
    setup_firewall
    create_config
    create_directories
    setup_python_env
    setup_redis
    setup_ssl
    setup_nginx
    setup_monitoring
    setup_log_rotation
    create_systemd_service
    setup_backup_scripts
    start_services
    health_check
    show_final_info
    
    log "Installation completed successfully!"
}

# Run main function with all arguments
main "$@"
