# System Design Case Studies: Learning from FAANG and Major Tech Companies

## Table of Contents
1. [TinyURL - URL Shortening Service](#1-tinyurl---url-shortening-service)
2. [Netflix - Video Streaming Platform](#2-netflix---video-streaming-platform)
3. [Amazon - E-commerce & Cloud Platform](#3-amazon---e-commerce--cloud-platform)
4. [Facebook/Meta - Social Media Platform](#4-facebookmeta---social-media-platform)
5. [Google Maps - Location and Navigation Service](#5-google-maps---location-and-navigation-service)
6. [Airbnb - Accommodation Marketplace](#6-airbnb---accommodation-marketplace)
7. [Twitter/X - Microblogging Platform](#7-twitterx---microblogging-platform)
8. [Uber - Ride-hailing Service](#8-uber---ride-hailing-service)
9. [WhatsApp - Messaging Service](#9-whatsapp---messaging-service)
10. [Zoom - Video Conferencing Platform](#10-zoom---video-conferencing-platform)
11. [Notification Service System](#11-notification-service-system)

---

## 1. TinyURL - URL Shortening Service

### **System Overview**
TinyURL is a URL shortening service that creates shorter aliases for long URLs, enabling users to share links more efficiently while tracking analytics and managing redirections.

### **Key Requirements**
- **Functional**: Generate unique short URLs, redirect to original URLs, custom aliases, expiration handling
- **Non-functional**: Handle millions of URLs, 100:1 read/write ratio, low latency (<100ms), high availability

### **Architecture Components**

#### **Core Services**
- **URL Generation Service**: Creates unique short codes using Base62 encoding
- **Redirection Service**: Handles GET requests and redirects users
- **Analytics Service**: Tracks click metrics and user behavior
- **Cache Layer**: Redis for frequently accessed URLs

#### **Data Storage**
NoSQL databases like MongoDB or Cassandra are preferred due to their ability to handle billions of simple key-value lookups and provide high scalability.

**Database Schema**:
```sql
URLs Table:
- short_url (PK)
- original_url
- user_id
- created_at
- expires_at
- click_count

Users Table:
- user_id (PK)
- email
- api_key
- created_at
```

#### **URL Generation Strategies**
1. **Counter-based**: Sequential ID converted to Base62
2. **Hash-based**: MD5/SHA-256 hash of original URL
3. **Random Generation**: Cryptographically secure random strings

### **Scalability Solutions**
- **Sharding**: Distribute URLs across multiple database instances
- **Caching**: Store popular URLs in Redis/Memcached
- **CDN**: Geographic distribution for faster access
- **Load Balancing**: Distribute traffic across multiple application servers

### **Key Learnings**
- Base62 encoding provides URL-safe, compact representations
- Caching is crucial for read-heavy workloads
- Counter-based generation ensures uniqueness but requires coordination
- Analytics processing should be asynchronous to avoid blocking redirections

---

## 2. Netflix - Video Streaming Platform

### **System Overview**
Netflix serves millions of users globally with high-quality video streaming, requiring systems that prioritize availability, scalability, and global performance.

### **Key Requirements**
- **Functional**: Video upload/storage, streaming, user recommendations, content search
- **Non-functional**: 99.99% uptime, global CDN, adaptive bitrate streaming, handle millions of concurrent users

### **Architecture Components**

#### **Microservices Architecture**
- **User Service**: Authentication, profiles, preferences
- **Content Service**: Video metadata, encoding, storage
- **Recommendation Engine**: ML-based content suggestions
- **Streaming Service**: Video delivery and playback
- **Billing Service**: Subscription management

#### **Content Delivery**
Netflix uses Open Connect CDN with appliances distributed across ISP networks globally, bringing content geographically closer to users to minimize latency and buffering.

#### **Data Storage**
- **Cassandra**: User viewing history, recommendations
- **MySQL**: Billing, user accounts
- **S3**: Video file storage
- **Elasticsearch**: Content search and discovery

#### **Video Processing Pipeline**
1. **Upload**: Content ingestion and validation
2. **Encoding**: Multiple formats and bitrates for adaptive streaming
3. **Storage**: Distributed across global regions
4. **CDN Distribution**: Pre-positioning popular content

### **Scalability Solutions**
- **Auto-scaling**: Dynamic resource allocation based on demand
- **Circuit Breakers**: Prevent cascading failures
- **Chaos Engineering**: Proactive failure testing
- **Regional Isolation**: Separate deployments per geographic region

### **Key Learnings**
- Availability trumps consistency for streaming services
- Predictive caching improves user experience
- Microservices enable independent scaling and deployment
- Global CDN is essential for streaming performance

---

## 3. Amazon - E-commerce & Cloud Platform

### **System Overview**
Amazon operates as both a massive e-commerce platform and cloud infrastructure provider, requiring systems that handle complex transactions, inventory management, and service orchestration.

### **Key Requirements**
- **Functional**: Product catalog, order processing, payments, inventory, recommendations
- **Non-functional**: 99.95% availability, handle Black Friday traffic spikes, global presence

### **Architecture Components**

#### **Service-Oriented Architecture (SOA)**
Each component—from product listings to payment processing—runs as an independent service, enabling independent scaling and deployment.

- **Catalog Service**: Product information and search
- **Cart Service**: Shopping cart management
- **Order Service**: Order processing and fulfillment
- **Payment Service**: Transaction processing
- **Inventory Service**: Stock management
- **Recommendation Service**: Product suggestions

#### **Data Management**
- **DynamoDB**: Product catalog, user sessions
- **RDS**: Order history, financial transactions
- **Redshift**: Analytics and business intelligence
- **S3**: Product images, static content

#### **Infrastructure Services**
- **API Gateway**: Request routing and rate limiting
- **Lambda**: Serverless compute for event processing
- **SQS/SNS**: Asynchronous messaging
- **CloudFront**: Global content delivery

### **Scalability Solutions**
- **Horizontal partitioning**: Shard data by customer/region
- **Event-driven architecture**: Asynchronous processing
- **Auto-scaling groups**: Dynamic capacity management
- **Multi-region deployment**: Geographic distribution

### **Key Learnings**
- SOA enables independent service evolution
- Event-driven patterns improve resilience
- Inventory systems require careful consistency management
- Payment processing needs strong consistency guarantees

---

## 4. Facebook/Meta - Social Media Platform

### **System Overview**
Facebook's News Feed and Messenger systems handle massive social interactions using event-driven architecture and graph-based data management.

### **Key Requirements**
- **Functional**: News feed, messaging, friend connections, content sharing
- **Non-functional**: Handle 3+ billion users, real-time updates, global presence

### **Architecture Components**

#### **News Feed Architecture**
The News Feed is built around an event-driven system where updates trigger downstream processes that decide which content to surface for each user.

- **Fan-out Service**: Distributes posts to followers' feeds
- **Ranking Service**: ML-based content prioritization
- **Timeline Service**: Manages user feed generation
- **Media Service**: Image/video processing and storage

#### **Messaging System**
- **Real-time Communication**: WebSocket connections for instant messaging
- **Message Storage**: Distributed across multiple data centers
- **Delivery Service**: Ensures message reliability
- **Presence Service**: Online status tracking

#### **Data Storage**
User relationships and content engagements form a massive social graph using sharded and distributed database solutions.

- **TAO**: Distributed graph database for social connections
- **MySQL**: Sharded for various data types
- **Memcached**: Distributed caching layer
- **Haystack**: Photo storage system

### **Scalability Solutions**
- **Graph partitioning**: Distribute social connections efficiently
- **Push/Pull hybrid**: Balance between read and write performance
- **Consistent hashing**: Distribute cache and data evenly
- **Edge timeline**: Pre-computed feeds for popular users

### **Key Learnings**
- Social graphs require specialized data structures
- Real-time systems need sophisticated message ordering
- Caching strategies must handle social locality
- Event-driven architecture enables real-time updates

---

## 5. Google Maps - Location and Navigation Service

### **System Overview**
Google Maps provides real-time navigation, traffic data, and location services to millions of users worldwide, requiring sophisticated geospatial data processing.

### **Key Requirements**
- **Functional**: Route calculation, real-time traffic, location search, navigation
- **Non-functional**: Sub-second response times, handle millions of queries, GPS accuracy

### **Architecture Components**

#### **Core Services**
- **Routing Service**: Calculates optimal paths using graph algorithms
- **Traffic Service**: Real-time traffic data collection and analysis
- **Geocoding Service**: Address to coordinate conversion
- **Tile Service**: Map rendering and caching
- **Location Service**: GPS tracking and positioning

#### **Data Management**
- **Geospatial Databases**: PostGIS for location data
- **Graph Databases**: Road network representation
- **Time-series Databases**: Traffic pattern storage
- **Distributed Cache**: Map tiles and route caching

#### **Real-time Processing**
- **Stream Processing**: Live traffic data from mobile devices
- **Machine Learning**: Traffic prediction models
- **Graph Algorithms**: Dijkstra's, A* for pathfinding
- **Spatial Indexing**: R-trees for efficient location queries

### **Scalability Solutions**
- **Geosharding**: Partition data by geographic regions
- **CDN**: Cache map tiles globally
- **Edge computing**: Process location data near users
- **Precomputed routes**: Store popular route calculations

### **Key Learnings**
- Geospatial data requires specialized indexing strategies
- Real-time traffic processing needs stream computing
- Map tile caching significantly improves performance
- Route calculation benefits from precomputation

---

## 6. Airbnb - Accommodation Marketplace

### **System Overview**
Airbnb operates as a peer-to-peer marketplace platform connecting hosts and guests, requiring systems that handle booking transactions, search, and trust mechanisms.

### **Key Requirements**
- **Functional**: Property search, booking system, payments, reviews, messaging
- **Non-functional**: Global availability, handle peak travel seasons, payment security

### **Architecture Components**

#### **Marketplace Services**
- **Search Service**: Property discovery with filters
- **Booking Service**: Reservation management
- **Payment Service**: Secure transaction processing
- **Review Service**: Trust and reputation system
- **Messaging Service**: Host-guest communication

#### **Data Architecture**
Originally built on Ruby on Rails, later scaled using services like Node.js and Java with React for frontend experiences.

- **PostgreSQL**: Booking and user data
- **Elasticsearch**: Property search functionality
- **Redis**: Session storage and caching
- **S3**: Property images and documents

#### **Search and Discovery**
- **Elasticsearch**: Full-text search with geolocation
- **ML Ranking**: Personalized search results
- **Pricing Engine**: Dynamic pricing recommendations
- **Availability Service**: Real-time booking status

### **Scalability Solutions**
- **Service decomposition**: Break monolith into microservices
- **Database sharding**: Partition by geographic regions
- **CDN**: Global image and content delivery
- **Async processing**: Background jobs for notifications

### **Key Learnings**
- Marketplace platforms need sophisticated search capabilities
- Trust systems require careful data modeling
- Payment processing needs regulatory compliance
- Geographic sharding aligns with user behavior patterns

---

## 7. Twitter/X - Microblogging Platform

### **System Overview**
Twitter handles massive volumes of real-time content distribution, requiring systems optimized for high write throughput and instant content delivery.

### **Key Requirements**
- **Functional**: Tweet posting, timeline generation, following/followers, trending topics
- **Non-functional**: Handle 500M tweets/day, real-time delivery, global presence

### **Architecture Components**

#### **Core Services**
- **Tweet Service**: Content creation and storage
- **Timeline Service**: Feed generation and delivery
- **Graph Service**: Follow relationships
- **Trending Service**: Real-time topic detection
- **Search Service**: Content discovery

#### **Timeline Generation**
- **Push Model**: Pre-compute timelines (fan-out on write)
- **Pull Model**: Generate timelines on request (fan-out on read)
- **Hybrid Approach**: Push for normal users, pull for celebrities

#### **Data Storage**
- **MySQL**: User data and relationships
- **Redis**: Timeline caches
- **Manhattan**: Distributed key-value store
- **Hadoop**: Analytics and batch processing

### **Scalability Solutions**
- **Fan-out strategies**: Optimize timeline delivery
- **Celebrity problem**: Special handling for high-follower accounts
- **Read replicas**: Scale read operations
- **Message queues**: Asynchronous timeline updates

### **Key Learnings**
- Timeline generation strategies significantly impact performance
- Celebrity users require special architectural considerations
- Real-time trending requires stream processing
- Write-heavy workloads need careful optimization

---

## 8. Uber - Ride-hailing Service

### **System Overview**
Uber's architecture handles real-time matching of riders and drivers, requiring systems that process location data, optimize routes, and manage dynamic pricing.

### **Key Requirements**
- **Functional**: Ride matching, real-time tracking, payments, driver management
- **Non-functional**: Sub-second matching, handle millions of rides/day, global scalability

### **Architecture Components**

#### **Core Services**
- **Matching Service**: Pair riders with nearby drivers
- **Location Service**: Real-time GPS tracking
- **Pricing Service**: Dynamic fare calculation
- **Trip Service**: Ride state management
- **Payment Service**: Transaction processing

#### **Real-time Processing**
- **Geospatial Indexing**: QuadTrees for efficient location queries
- **Stream Processing**: Real-time location updates
- **Machine Learning**: Demand prediction and pricing
- **Graph Processing**: Route optimization

#### **Data Management**
- **Cassandra**: Location and trip data
- **PostgreSQL**: User and driver information
- **Redis**: Real-time cache for active rides
- **Kafka**: Event streaming platform

### **Scalability Solutions**
- **Geosharding**: Partition by city/region
- **Supply-demand balancing**: Dynamic resource allocation
- **Microservices**: Independent service scaling
- **Event sourcing**: Audit trail for trip events

### **Key Learnings**
- Location-based services require specialized data structures
- Real-time matching needs sophisticated algorithms
- Dynamic pricing requires careful economic modeling
- Geospatial sharding improves query performance

---

## 9. WhatsApp - Messaging Service

### **System Overview**
WhatsApp delivers billions of messages daily with minimal infrastructure, emphasizing simplicity and reliability in message delivery.

### **Key Requirements**
- **Functional**: Message delivery, group chats, media sharing, end-to-end encryption
- **Non-functional**: Handle 100B messages/day, low latency, high availability

### **Architecture Components**

#### **Message Processing**
- **Message Router**: Distributes messages to recipients
- **Delivery Service**: Ensures message reliability
- **Encryption Service**: End-to-end security
- **Media Service**: Image/video handling
- **Presence Service**: Online status tracking

#### **Infrastructure**
- **Erlang/OTP**: Concurrent message processing
- **FreeBSD**: Operating system optimization
- **Custom protocols**: Efficient mobile communication
- **Minimal dependencies**: Reduced complexity

#### **Data Storage**
- **In-memory storage**: Active conversation data
- **Distributed databases**: Message history
- **Media storage**: Compressed file handling
- **Backup systems**: Message recovery

### **Scalability Solutions**
- **Actor model**: Erlang's lightweight processes
- **Connection pooling**: Efficient resource utilization
- **Message queuing**: Asynchronous delivery
- **Horizontal scaling**: Add servers as needed

### **Key Learnings**
- Simplicity in architecture improves reliability
- Erlang excels at concurrent message processing
- Mobile optimization requires protocol efficiency
- End-to-end encryption can be implemented at scale

---

## 10. Zoom - Video Conferencing Platform

### **System Overview**
Zoom provides high-quality video conferencing with features like screen sharing, recording, and large-scale meetings, requiring optimized media processing.

### **Key Requirements**
- **Functional**: Video/audio calls, screen sharing, recording, chat, webinars
- **Non-functional**: Support 1000+ participants, low latency, cross-platform

### **Architecture Components**

#### **Media Processing**
- **Media Router**: Routes audio/video streams
- **Transcoding Service**: Format conversion for different devices
- **Recording Service**: Session capture and storage
- **Screen Sharing**: Desktop streaming optimization
- **Quality Control**: Adaptive bitrate and bandwidth management

#### **Signaling and Control**
- **Session Management**: Call state and participant tracking
- **Authentication Service**: User verification and permissions
- **Scheduling Service**: Meeting planning and invitations
- **Chat Service**: Text messaging during calls

#### **Infrastructure**
- **WebRTC**: Browser-based real-time communication
- **UDP**: Low-latency media transport
- **CDN**: Global media distribution
- **Cloud deployment**: Multi-region presence

### **Scalability Solutions**
- **Media server clustering**: Distributed processing
- **Adaptive streaming**: Quality adjustment based on bandwidth
- **Regional deployment**: Reduce latency through proximity
- **Load balancing**: Distribute call processing

### **Key Learnings**
- Video quality requires adaptive streaming strategies
- WebRTC enables browser-based communication
- Low latency is critical for real-time interaction
- Recording systems need efficient storage and retrieval

---

## 11. Notification Service System

### **System Overview**
A notification service delivers timely information to users across various channels including SMS, email, push notifications, and in-app messages.

### **Key Requirements**
- **Functional**: Multi-channel support, user preferences, scheduling, templates
- **Non-functional**: Handle millions of notifications/day, reliable delivery, low latency

### **Architecture Components**

#### **Core Services**
The Notification Service creates and formats messages for required channels, placing each message into respective topics in the Notification Queue System.

- **Notification Service**: Message creation and formatting
- **Template Service**: Dynamic content generation
- **User Preference Service**: Opt-in/out management
- **Analytics Service**: Delivery tracking and metrics

#### **Channel Processors**
Channel Processors are responsible for pulling notifications from the Notification Queue and delivering them to users via specific channels, enabling independent scaling and asynchronous processing.

- **Email Processor**: SMTP integration
- **SMS Processor**: Telecom gateway integration
- **Push Processor**: Mobile notification services
- **In-app Processor**: WebSocket-based delivery

#### **Queue System**
Each channel has its own dedicated topic, ensuring that messages are processed independently by relevant Channel Processors.

- **Message Queues**: Kafka/RabbitMQ for reliable delivery
- **Dead Letter Queues**: Failed message handling
- **Priority Queues**: Urgent vs. normal notifications
- **Retry Logic**: Automatic failure recovery

### **Scalability Solutions**
- **Channel separation**: Independent processing per delivery method
- **Rate limiting**: Respect external service limits
- **Batch processing**: Optimize throughput for bulk sends
- **Circuit breakers**: Handle external service failures

### **Key Learnings**
- Multi-channel delivery requires independent processing pipelines
- User preferences must be checked before every delivery
- Rate limiting prevents external service throttling
- Analytics help optimize delivery timing and content

---

## Summary of Key Architectural Patterns

### **Common Patterns Across All Systems**
1. **Microservices Architecture**: Enables independent scaling and deployment
2. **Event-Driven Design**: Improves system resilience and responsiveness
3. **Caching Strategies**: Critical for read-heavy workloads
4. **Database Sharding**: Horizontal scaling for large datasets
5. **Load Balancing**: Distributes traffic across multiple servers
6. **Asynchronous Processing**: Improves user experience and system throughput

### **Technology Choices by Use Case**
- **Read-Heavy**: Redis, CDN, read replicas
- **Write-Heavy**: Kafka, distributed databases, async processing
- **Real-time**: WebSockets, stream processing, in-memory databases
- **Global Scale**: Multi-region deployment, edge computing, CDN
- **High Availability**: Circuit breakers, redundancy, chaos engineering

### **Trade-offs to Consider**
- **Consistency vs. Availability**: CAP theorem implications
- **Cost vs. Performance**: Resource optimization strategies
- **Complexity vs. Maintainability**: Architecture evolution paths
- **Security vs. Performance**: Authentication and encryption overhead
- **Latency vs. Throughput**: System optimization priorities

This comprehensive study of these 11 major systems provides insights into how different architectural decisions impact system behavior at scale. Each company has made specific trade-offs based on their unique requirements, user base, and business constraints.