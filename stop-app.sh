#!/bin/bash

# Change to the prj-docker-compose directory where docker-compose.yaml is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/prj-docker-compose" || { echo "Error: Cannot change to prj-docker-compose directory"; exit 1; }

# Function to display help
show_help() {
  echo "Usage: $0 [--remove-volumes|-v] [--remove-images|-i] [--clean-all|-ca] [--help|-h]"
  echo ""
  echo "Options:"
  echo "  --remove-volumes, -v  : Stop containers and remove volumes"
  echo "  --remove-images, -i   : Stop containers and remove images"
  echo "  --clean-all, -ca      : Stop containers, remove volumes, images, and network"
  echo "  --help, -h            : Show this help message"
  echo ""
  echo "Default behavior:"
  echo "  Stops containers without removing volumes or images"
  echo ""
  echo "Examples:"
  echo "  $0                     Stop containers (keep volumes)"
  echo "  $0 --remove-volumes    Stop containers and remove volumes"
  echo "  $0 --remove-images     Stop containers and remove images"
  echo "  $0 --clean-all         Complete cleanup (containers, volumes, images, network)"
  echo ""
  echo "WARNING: --remove-volumes will delete all database data!"
  echo "WARNING: --remove-images will delete all built images!"
  exit 0
}

# Process arguments
REMOVE_VOLUMES=false
REMOVE_IMAGES=false
CLEAN_ALL=false

for arg in "$@"; do
  case "$arg" in
    --help|-h)
      show_help
      ;;
    --remove-volumes|-v)
      REMOVE_VOLUMES=true
      ;;
    --remove-images|-i)
      REMOVE_IMAGES=true
      ;;
    --clean-all|-ca)
      CLEAN_ALL=true
      ;;
    *)
      echo "Error: Unknown parameter '$arg'"
      echo "Use '$0 --help' to see the complete help"
      exit 1
      ;;
  esac
done

# Check for running containers
check_containers() {
  local running_services=$(docker compose ps --services --filter "status=running" 2>/dev/null)
  
  if [ -z "$running_services" ]; then
    echo "No containers are currently running."
    return 1
  fi
  
  echo "Stopping the following services:"
  echo "$running_services"
  echo ""
  return 0
}

# Stop log capture if running
stop_log_capture() {
  local pid_file="$SCRIPT_DIR/logs/app_complete-logger.pid"
  
  if [ -f "$pid_file" ]; then
    local pid=$(cat "$pid_file" 2>/dev/null)
    if [ -n "$pid" ] && ps -p "$pid" > /dev/null 2>&1; then
      echo "Stopping log capture (PID: $pid)..."
      kill "$pid" 2>/dev/null || true
    fi
    rm -f "$pid_file"
  fi
}

# Display header
echo "=================================================================="
echo "              STOPPING APPLICATION                     "
echo "=================================================================="
echo ""

# Stop log capture
stop_log_capture

# Check if containers are running (skip check for clean-all operations)
if ! check_containers; then
  # For clean-all, continue anyway to remove orphaned images and networks
  if [ "$CLEAN_ALL" = true ]; then
    echo "No containers are currently running, but continuing cleanup..."
    echo ""
  elif [ "$REMOVE_IMAGES" = true ]; then
    echo "No containers are currently running, but continuing image removal..."
    echo ""
  else
    exit 0
  fi
fi

# Perform stop operation based on options
if [ "$CLEAN_ALL" = true ]; then
  echo "🔄 Performing complete cleanup..."
  docker compose down -v --remove-orphans
  
  # Remove images
  echo "🗑️  Removing images..."
  docker rmi ai_agents_hospitality-api bookings-db-data-loader postgres:15.3 2>/dev/null || true
  
  # Remove custom network (it should be removed by down, but just in case)
  echo "🗑️  Removing network..."
  docker network rm prj-docker-compose_prj_hospitality-network 2>/dev/null || true
  
  echo ""
  echo "✅ Complete cleanup finished!"
  echo "   - Containers removed"
  echo "   - Volumes removed"
  echo "   - Images removed"
  echo "   - Network removed"
  
elif [ "$REMOVE_VOLUMES" = true ] && [ "$REMOVE_IMAGES" = true ]; then
  echo "🔄 Stopping containers, removing volumes and images..."
  docker compose down -v --remove-orphans
  
  echo "🗑️  Removing images..."
  docker rmi ai_agents_hospitality-api bookings-db-data-loader postgres:15.3 2>/dev/null || true
  
  echo ""
  echo "✅ Containers stopped, volumes and images removed!"
  
elif [ "$REMOVE_VOLUMES" = true ]; then
  echo "🔄 Stopping containers and removing volumes..."
  docker compose down -v --remove-orphans
  
  echo ""
  echo "⚠️  WARNING: All database data has been removed!"
  echo "✅ Containers stopped and volumes removed!"
  
elif [ "$REMOVE_IMAGES" = true ]; then
  echo "🔄 Stopping containers and removing images..."
  docker compose down --remove-orphans
  
  echo "🗑️  Removing images..."
  docker rmi ai_agents_hospitality-api bookings-db-data-loader postgres:15.3 2>/dev/null || true
  
  echo ""
  echo "✅ Containers stopped and images removed!"
  
else
  echo "🛑 Stopping containers (keeping volumes)..."
  docker compose down --remove-orphans
  
  echo ""
  echo "✅ Containers stopped successfully!"
  echo "💾 Volumes preserved (data retained)"
fi

# Show final status
echo ""
echo "📊 Final Status:"
docker compose ps
echo ""

echo "=================================================================="
echo "                    STOP OPERATION COMPLETE                       "
echo "=================================================================="
echo ""
echo "To start the application again, use:"
echo "  ./start-app.sh"
echo ""

