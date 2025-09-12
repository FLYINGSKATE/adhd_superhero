# 🚀 System Design Practice: 32 MCQs + 10 Real-World Interview Questions

> 🎯 **Test Your Knowledge** and ace those system design interviews like a pro! 💪

## 📝 Part A: Multiple Choice Questions (32 Questions)

### **🏗️ Section 1: Scalability and Architecture (Questions 1-8)**

**1. 🔗 Which approach is most suitable for handling 100:1 read-to-write ratio in a URL shortening service?**
- A) Use only write replicas 📝
- B) Implement aggressive caching with read replicas ⚡
- C) Use only master-slave replication 🔄
- D) Store everything in memory 💾

**✅ Answer: B) Implement aggressive caching with read replicas**
*💡 Explanation: Read-heavy workloads benefit most from caching frequently accessed data and using read replicas to distribute query load.*

**2. 🎬 What is the primary advantage of Netflix using a microservices architecture?**
- A) Reduces total cost of ownership 💰
- B) Simplifies debugging and monitoring 🔍
- C) Enables independent scaling and deployment of services 🚀
- D) Eliminates the need for load balancers ⚖️

**✅ Answer: C) Enables independent scaling and deployment of services**
*💡 Explanation: Microservices allow teams to scale and deploy services independently based on specific demand patterns.*

**3. 👥 Which database type is most appropriate for storing social graph relationships in Facebook's architecture?**
- A) Relational database with foreign keys 🔗
- B) Document-oriented NoSQL database 📄
- C) Graph database optimized for relationship queries 🕸️
- D) Time-series database ⏰

**✅ Answer: C) Graph database optimized for relationship queries**
*💡 Explanation: Graph databases are specifically designed to efficiently store and query relationship data like social connections.*

**4. 🔄 What is the main reason for using consistent hashing in distributed systems?**
- A) To improve data consistency across nodes 🔐
- B) To minimize data movement when nodes are added or removed ⚡
- C) To encrypt data during transmission 🛡️
- D) To compress data for storage efficiency 📦

**✅ Answer: B) To minimize data movement when nodes are added or removed**
*💡 Explanation: Consistent hashing reduces the amount of data that needs to be redistributed when the cluster topology changes.*

**5. 🐦 Which strategy best addresses the "celebrity problem" in Twitter's timeline generation?**
- A) Use only push-based (fan-out on write) approach 📤
- B) Use only pull-based (fan-out on read) approach 📥
- C) Implement a hybrid approach with different strategies for different user types 🔄
- D) Cache all celebrity tweets permanently 💾

**✅ Answer: C) Implement a hybrid approach with different strategies for different user types**
*💡 Explanation: Hybrid approaches use push for normal users and pull for celebrities to avoid overwhelming the system with fan-out operations.*

**6. 🎥 What is the primary benefit of using a CDN in Netflix's architecture?**
- A) Reduces storage costs 💰
- B) Improves video encoding quality 🎬
- C) Brings content geographically closer to users 🌍
- D) Eliminates the need for load balancers ⚖️

**✅ Answer: C) Brings content geographically closer to users**
*💡 Explanation: CDNs cache content at edge locations worldwide, reducing latency and improving streaming performance.*

**7. 🚗 Which approach is most effective for handling geospatial queries in Uber's location service?**
- A) Store coordinates as strings in a relational database 📝
- B) Use spatial indexing with data structures like QuadTrees 🗺️
- C) Process all location data in real-time without indexing ⚡
- D) Use only in-memory storage for all location data 💾

**✅ Answer: B) Use spatial indexing with data structures like QuadTrees**
*💡 Explanation: Spatial indexing structures like QuadTrees or R-trees enable efficient location-based queries for nearby drivers.*

**8. 🛒 What is the main advantage of Amazon's Service-Oriented Architecture (SOA)?**
- A) Reduces network latency between services ⚡
- B) Eliminates the need for data replication 🔄
- C) Enables independent evolution and scaling of business capabilities 🚀
- D) Simplifies database schema design 📊

**✅ Answer: C) Enables independent evolution and scaling of business capabilities**
*💡 Explanation: SOA allows different business functions to evolve independently while maintaining well-defined interfaces.*

### **🗄️ Section 2: Data Storage and Management (Questions 9-16)**

**9. 🔗 Why does TinyURL prefer NoSQL databases over relational databases?**
- A) NoSQL provides better data consistency 🔐
- B) NoSQL offers better horizontal scalability for simple key-value operations 📈
- C) NoSQL has better SQL query support 🔍
- D) NoSQL provides automatic data compression 📦

**✅ Answer: B) NoSQL offers better horizontal scalability for simple key-value operations**
*💡 Explanation: URL shortening primarily involves simple key-value lookups, which NoSQL databases handle more efficiently at scale.*

**10. 💳 Which database property is most important for Uber's payment processing system?**
- A) Eventual consistency 🔄
- B) High availability over consistency 🔼
- C) Strong consistency (ACID properties) 🛡️
- D) Partition tolerance over availability 🔀

**✅ Answer: C) Strong consistency (ACID properties)**
*💡 Explanation: Financial transactions require strong consistency to ensure accurate payment processing and prevent double-spending.*

**11. 💬 What is the primary reason WhatsApp uses Erlang for message processing?**
- A) Erlang provides built-in encryption 🔐
- B) Erlang excels at handling massive concurrency with lightweight processes ⚡
- C) Erlang has the fastest database drivers 🏎️
- D) Erlang automatically scales across multiple servers 📈

**✅ Answer: B) Erlang excels at handling massive concurrency with lightweight processes**
*💡 Explanation: Erlang's actor model and lightweight processes make it ideal for handling millions of concurrent connections.*

**12. ⚡ Which caching strategy is most appropriate for frequently accessed shortened URLs?**
- A) Write-through caching 📝
- B) Write-behind caching 🔄
- C) Cache-aside (lazy loading) 💤
- D) Distributed caching with TTL ⏰

**✅ Answer: D) Distributed caching with TTL**
*💡 Explanation: Popular URLs should be cached across multiple nodes with appropriate TTL to handle high read volumes.*

**13. 🏠 How does Airbnb handle property search across millions of listings efficiently?**
- A) Linear search through all properties 📝
- B) Use Elasticsearch with geolocation and filtering capabilities 🔍
- C) Store all data in memory for faster access 💾
- D) Use only SQL databases with complex joins 🔗

**✅ Answer: B) Use Elasticsearch with geolocation and filtering capabilities**
*💡 Explanation: Elasticsearch provides full-text search, geolocation queries, and complex filtering needed for property discovery.*

**14. 🔔 What is the primary challenge in designing notification service data storage?**
- A) Handling different message formats across multiple channels 📱
- B) Storing user preferences and managing opt-in/opt-out efficiently ⚙️
- C) Encrypting all notification content 🔐
- D) Compressing notification data for storage 📦

**✅ Answer: B) Storing user preferences and managing opt-in/opt-out efficiently**
*💡 Explanation: User preferences for different notification types and channels must be efficiently stored and quickly accessible.*

**15. 📹 Which data partitioning strategy is most effective for Zoom's video conferencing system?**
- A) Partition by user ID hash 👤
- B) Partition by meeting ID and geographic region 🌍
- C) Partition by timestamp ⏰
- D) Use no partitioning, store everything centrally 🏢

**✅ Answer: B) Partition by meeting ID and geographic region**
*💡 Explanation: Partitioning by meeting and region keeps related data together and reduces latency for participants.*

**16. 🚗 What is the main advantage of using event sourcing in Uber's trip management?**
- A) Reduces storage requirements 💾
- B) Provides complete audit trail and enables event replay 📜
- C) Simplifies database schema design 📊
- D) Eliminates the need for backup systems 💽

**✅ Answer: B) Provides complete audit trail and enables event replay**
*💡 Explanation: Event sourcing maintains a complete history of trip events, enabling auditing and system recovery.*

### **⚡ Section 3: Real-time Systems and Communication (Questions 17-24)**

**17. 💬 Which protocol is most suitable for real-time messaging in WhatsApp?**
- A) HTTP with long polling 🔄
- B) WebSockets for persistent connections 🔌
- C) REST API with frequent polling 📡
- D) FTP for file transfer 📁

**✅ Answer: B) WebSockets for persistent connections**
*💡 Explanation: WebSockets maintain persistent connections enabling real-time, bidirectional communication with minimal overhead.*

**18. 🗺️ How does Google Maps achieve sub-second route calculation for millions of users?**
- A) Use only real-time computation for all routes ⚡
- B) Precompute popular routes and use caching with graph algorithms 🧮
- C) Store all possible routes in advance 💾
- D) Use linear search through all road segments 📍

**✅ Answer: B) Precompute popular routes and use caching with graph algorithms**
*💡 Explanation: Combining precomputed routes for popular paths with real-time graph algorithms and caching provides optimal performance.*

**19. 📹 What is the primary challenge in Zoom's real-time video processing?**
- A) Video compression algorithms 🎥
- B) Managing bandwidth adaptation and latency optimization ⚡
- C) User interface design 🎨
- D) File storage optimization 💾

**✅ Answer: B) Managing bandwidth adaptation and latency optimization**
*💡 Explanation: Real-time video requires adaptive bitrate streaming and latency optimization for quality user experience.*

**20. 🔥 Which approach best handles Twitter's real-time trending topics calculation?**
- A) Batch processing every hour ⏰
- B) Stream processing with sliding window algorithms 🌊
- C) Manual curation by content moderators 👥
- D) Simple counting without time considerations 🔢

**✅ Answer: B) Stream processing with sliding window algorithms**
*💡 Explanation: Real-time trending requires continuous processing of tweet streams with time-windowed analysis.*

**21. 📰 How does Facebook ensure real-time news feed updates for billions of users?**
- A) Periodic polling every minute ⏰
- B) Event-driven architecture with push notifications and fan-out 📡
- C) Manual refresh by users 🔄
- D) Batch processing every few hours ⏳

**✅ Answer: B) Event-driven architecture with push notifications and fan-out**
*💡 Explanation: Event-driven systems with intelligent fan-out strategies enable real-time updates at massive scale.*

**22. 🎬 What is the main requirement for Netflix's video streaming to handle millions of concurrent viewers?**
- A) Single high-performance server 🖥️
- B) Adaptive bitrate streaming with global CDN 🌍
- C) Store all videos in a single data center 🏢
- D) Use only the highest quality video format 🎥

**✅ Answer: B) Adaptive bitrate streaming with global CDN**
*💡 Explanation: Adaptive streaming adjusts quality based on user bandwidth while CDN ensures global content availability.*

**23. 🔔 Which messaging pattern is most appropriate for notification service delivery?**
- A) Synchronous request-response 🔄
- B) Asynchronous message queues with retry logic 📬
- C) Direct database polling 🔍
- D) File-based communication 📁

**✅ Answer: B) Asynchronous message queues with retry logic**
*💡 Explanation: Asynchronous queues decouple services and provide reliability through retry mechanisms for failed deliveries.*

**24. 🚗 How does Uber achieve real-time driver-rider matching within seconds?**
- A) Check all drivers sequentially 📝
- B) Use geospatial indexing with proximity algorithms 📍
- C) Random assignment within city limits 🎲
- D) Manual dispatcher assignment 👨‍💼

**✅ Answer: B) Use geospatial indexing with proximity algorithms**
*💡 Explanation: Spatial data structures enable quick location queries to find nearby drivers efficiently.*

### **⚖️ Section 4: System Design Principles and Trade-offs (Questions 25-32)**

**25. 🔺 According to the CAP theorem, what trade-off must distributed systems make?**
- A) Choose between speed and accuracy ⚡
- B) Balance consistency, availability, and partition tolerance (can't have all three) ⚖️
- C) Optimize for either read or write performance 📖
- D) Select between security and performance 🔐

**✅ Answer: B) Balance consistency, availability, and partition tolerance (can't have all three)**
*💡 Explanation: CAP theorem states that distributed systems can only guarantee two of the three properties simultaneously.*

**26. 🛒 What is the primary trade-off when implementing Amazon's eventual consistency model?**
- A) Higher cost vs. lower performance 💰
- B) Immediate consistency vs. higher availability and partition tolerance ⚖️
- C) Security vs. usability 🔐
- D) Complexity vs. simplicity 🧩

**✅ Answer: B) Immediate consistency vs. higher availability and partition tolerance**
*💡 Explanation: Eventual consistency sacrifices immediate consistency to achieve better availability and fault tolerance.*

**27. 💬 Which principle guides WhatsApp's architecture of handling 100 billion messages daily with minimal infrastructure?**
- A) Over-engineering for future requirements 🏗️
- B) Complexity through advanced algorithms 🧮
- C) Simplicity and efficiency in design ✨
- D) Using the latest technology trends 🆕

**✅ Answer: C) Simplicity and efficiency in design**
*💡 Explanation: WhatsApp's success comes from focusing on core functionality with simple, efficient architecture.*

**28. 🔧 What is the main reason for implementing circuit breakers in Netflix's microservices?**
- A) Improve system performance ⚡
- B) Prevent cascading failures across services 🛡️
- C) Reduce network bandwidth usage 📊
- D) Simplify service deployment 🚀

**✅ Answer: B) Prevent cascading failures across services**
*💡 Explanation: Circuit breakers stop requests to failing services, preventing failures from propagating through the system.*

**29. 🔗 Why does TinyURL use Base62 encoding for short URLs instead of Base64?**
- A) Base62 provides better compression 📦
- B) Base62 is faster to compute ⚡
- C) Base62 uses URL-safe characters (avoids +, /, =) ✅
- D) Base62 provides better security 🔐

**✅ Answer: C) Base62 uses URL-safe characters (avoids +, /, =)**
*💡 Explanation: Base62 uses only alphanumeric characters, making URLs safe for web usage without encoding issues.*

**30. 🏠 What is the primary consideration when designing Airbnb's global payment processing system?**
- A) Processing speed over accuracy ⚡
- B) Regulatory compliance and multi-currency support 🌍
- C) Minimizing transaction fees 💰
- D) Supporting only credit card payments 💳

**✅ Answer: B) Regulatory compliance and multi-currency support**
*💡 Explanation: Global payment systems must comply with various financial regulations and handle multiple currencies.*

**31. 🗺️ Which approach best balances performance and cost in Google Maps' tile serving system?**
- A) Generate all tiles in real-time ⚡
- B) Pre-generate and cache tiles at multiple zoom levels with CDN 🎯
- C) Store only the highest resolution tiles 🔍
- D) Use vector graphics for all map rendering 📐

**✅ Answer: B) Pre-generate and cache tiles at multiple zoom levels with CDN**
*💡 Explanation: Pre-generated tiles with CDN caching provide fast access while managing storage costs effectively.*

**32. 📹 What is the key architectural decision that enables Zoom to support 1000+ participants in a single meeting?**
- A) Use peer-to-peer connections between all participants 🤝
- B) Implement selective forwarding units (SFUs) with media routing 🎛️
- C) Require all participants to have high-bandwidth connections 📡
- D) Limit video quality to reduce bandwidth requirements 📉

**✅ Answer: B) Implement selective forwarding units (SFUs) with media routing**
*💡 Explanation: SFUs efficiently route media streams, reducing bandwidth requirements compared to mesh topology.*

---

## 🎯 Part B: Real-World Interview Questions (10 Questions)

### **🔗 Question 1: Design a URL Shortening Service (TinyURL)**
**🎬 Scenario**: You need to design a URL shortening service that can handle 100 million URLs being shortened per day, with a read-to-write ratio of 100:1.

**💭 Key Discussion Points**:
- 🎲 How would you generate unique short URLs? Compare different approaches (counter-based, hash-based, random generation)
- 🗄️ How would you handle the database design and what type of database would you choose?
- ⚡ What caching strategy would you implement for high read throughput?
- 📊 How would you handle analytics and tracking click metrics?
- 📈 What are the scalability bottlenecks and how would you address them?

**🔥 Follow-up**: How would you implement custom aliases and URL expiration?

---

### **🎬 Question 2: Design Netflix's Video Streaming Architecture**
**🎭 Scenario**: Design a global video streaming platform that can serve 100 million users worldwide with high-quality video content.

**💭 Key Discussion Points**:
- 🎥 How would you handle video storage, encoding, and delivery?
- 🌍 What is your CDN strategy for global content distribution?
- 📊 How would you implement adaptive bitrate streaming?
- 🤖 How would you design the recommendation system?
- 🚀 What are your strategies for handling peak traffic (e.g., popular show releases)?

**🔥 Follow-up**: How would you implement offline viewing and content pre-loading?

---

### **👥 Question 3: Design Facebook's News Feed System**
**🌐 Scenario**: Design a news feed system that can handle 3 billion users posting and consuming content in real-time.

**💭 Key Discussion Points**:
- 📡 How would you approach the fan-out problem (push vs. pull vs. hybrid)?
- ⭐ How would you handle the celebrity user problem?
- 🏆 What is your strategy for content ranking and personalization?
- 🗄️ How would you design the data model for posts, relationships, and feeds?
- ⚡ How would you ensure real-time updates?

**🔥 Follow-up**: How would you implement content moderation at scale?

---

### **🚗 Question 4: Design Uber's Ride Matching System**
**🌍 Scenario**: Design a system that can match millions of riders with drivers in real-time across multiple cities globally.

**💭 Key Discussion Points**:
- 📍 How would you handle real-time location tracking and updates?
- 🗺️ What data structures would you use for efficient proximity searches?
- 🤝 How would you implement the matching algorithm considering factors like distance, price, driver ratings?
- 💰 How would you handle high-demand scenarios (surge pricing)?
- 🚗 What is your approach to handling trip state management?

**🔥 Follow-up**: How would you implement ride sharing (UberPool) with multiple passengers?

---

### **💬 Question 5: Design WhatsApp's Messaging System**
**📱 Scenario**: Design a messaging system that can handle 100 billion messages per day with end-to-end encryption.

**💭 Key Discussion Points**:
- 🚚 How would you ensure message delivery reliability?
- 👻 What is your approach to handling online/offline users?
- 👥 How would you implement group messaging efficiently?
- 📱 How would you design the protocol for mobile optimization?
- 🖼️ What is your strategy for handling media messages (images, videos)?

**🔥 Follow-up**: How would you implement message synchronization across multiple devices?

---

### **🗺️ Question 6: Design Google Maps' Navigation System**
**🧭 Scenario**: Design a navigation system that provides real-time routing and traffic information to millions of users globally.

**💭 Key Discussion Points**:
- 🛣️ How would you model and store road network data?
- 🧮 What algorithms would you use for route calculation?
- 🚦 How would you collect and process real-time traffic data?
- 🔄 How would you handle route recalculation during navigation?
- 💾 What is your caching strategy for maps and routes?

**🔥 Follow-up**: How would you implement ETA prediction with machine learning?

---

### **🛒 Question 7: Design Amazon's Product Recommendation System**
**🎯 Scenario**: Design a recommendation system that can suggest relevant products to 300 million users based on their browsing and purchase history.

**💭 Key Discussion Points**:
- 🤖 What recommendation algorithms would you implement (collaborative filtering, content-based, hybrid)?
- 🆕 How would you handle the cold start problem for new users/products?
- 🌊 How would you design the data pipeline for real-time recommendations?
- ⚡ What is your approach to handling scale and serving recommendations quickly?
- 📊 How would you measure and improve recommendation quality?

**🔥 Follow-up**: How would you handle seasonal trends and trending products?

---

### **🏠 Question 8: Design Airbnb's Search and Booking System**
**🌍 Scenario**: Design a system that allows users to search for accommodations and make bookings, handling millions of properties and bookings globally.

**💭 Key Discussion Points**:
- 🔍 How would you design the search functionality with multiple filters (location, price, amenities)?
- 📅 How would you handle availability checking and booking conflicts?
- 🌏 What is your approach to handling different time zones and currencies?
- ⭐ How would you implement the review and rating system?
- 💰 How would you design the pricing engine for dynamic pricing?

**🔥 Follow-up**: How would you handle fraud detection and trust & safety?

---

### **🔔 Question 9: Design a Global Notification Service**
**📱 Scenario**: Design a notification service that can send notifications through multiple channels (email, SMS, push, in-app) to millions of users across different applications.

**💭 Key Discussion Points**:
- 📨 How would you design the system to support multiple notification channels?
- ⚙️ How would you handle user preferences and opt-in/opt-out management?
- 🛡️ What is your approach to ensuring delivery reliability and handling failures?
- 🚦 How would you implement rate limiting to respect external service limits?
- 🏢 How would you design the system for multi-tenancy (serving multiple client applications)?

**🔥 Follow-up**: How would you implement notification scheduling and batching?

---

### **📹 Question 10: Design Zoom's Video Conferencing System**
**🎥 Scenario**: Design a video conferencing system that can support meetings with up to 1000 participants with features like screen sharing and recording.

**💭 Key Discussion Points**:
- 🎛️ How would you handle video/audio streaming for large meetings?
- 📊 What is your approach to bandwidth optimization and quality adaptation?
- 🖥️ How would you implement screen sharing efficiently?
- 📡 How would you design the signaling system for meeting management?
- 📱 What is your strategy for handling different device types and network conditions?

**🔥 Follow-up**: How would you implement breakout rooms and meeting recording?

---

## 🎯 Answer Framework for Interview Questions

### **📋 1. Requirements Gathering (5 minutes)**
- **🔧 Functional Requirements**: What features does the system need?
- **⚡ Non-functional Requirements**: Scale, performance, availability requirements
- **🚧 Constraints**: Budget, timeline, technology restrictions

### **📊 2. Capacity Estimation (5 minutes)**
- **👥 Users**: Number of active users (daily/monthly)
- **💾 Data**: Storage requirements and growth projections
- **📡 Bandwidth**: Network requirements for different operations
- **⚡ QPS**: Queries per second calculations

### **🏗️ 3. System Design (15-20 minutes)**
- **🎯 High-level Architecture**: Major components and their interactions
- **🗄️ Database Design**: Schema, partitioning, consistency requirements
- **🔌 API Design**: Key endpoints and their functionality
- **🔍 Deep Dive**: Focus on 1-2 critical components

### **📈 4. Scale and Optimization (10 minutes)**
- **🚧 Bottlenecks**: Identify potential failure points
- **🚀 Scaling Strategy**: Horizontal vs. vertical scaling approaches
- **⚡ Caching**: What and where to cache
- **📊 Monitoring**: Key metrics and alerting

### **⚖️ 5. Trade-offs and Alternatives (5 minutes)**
- **🛠️ Technology Choices**: Justify your technology selections
- **🔄 Consistency vs. Availability**: CAP theorem considerations
- **💰 Performance vs. Cost**: Optimization trade-offs
- **🔐 Security**: Authentication, authorization, data protection

---

## 💡 Tips for Success

### **🎯 Before the Interview**
1. **📚 Study System Components**: Understand load balancers, databases, caching, message queues
2. **🧮 Practice Calculations**: Get comfortable with back-of-the-envelope estimations
3. **⚖️ Know Trade-offs**: Understand when to use different technologies and architectures
4. **🏢 Review Real Systems**: Study how major companies solve similar problems

### **🎪 During the Interview**
1. **❓ Ask Clarifying Questions**: Understand requirements before designing
2. **🎯 Start High-level**: Begin with overall architecture before diving into details
3. **💭 Think Out Loud**: Explain your reasoning and decision-making process
4. **⚖️ Consider Trade-offs**: Discuss pros and cons of different approaches
5. **✅ Be Realistic**: Acknowledge limitations and areas for improvement

### **❌ Common Mistakes to Avoid**
1. **🏃‍♂️ Jumping to Solutions**: Design without understanding requirements
2. **🏗️ Over-engineering**: Adding unnecessary complexity
3. **📉 Ignoring Scale**: Not considering performance at the required scale
4. **💥 Single Points of Failure**: Designing systems without redundancy
5. **👁️ Neglecting Monitoring**: Not considering observability and maintenance

> 💬 **Remember**: System design interviews are conversations, not exams! The interviewer wants to see how you think through complex problems, make trade-offs, and communicate technical concepts clearly. 🎯✨

### **🏆 Final Pro Tips**
- 🎨 **Draw diagrams** - Visual aids help everyone understand your design
- 🗣️ **Communicate clearly** - Explain your thought process step by step
- 🤔 **Ask questions** - Clarify requirements and constraints
- ⚖️ **Discuss trade-offs** - Show you understand engineering decisions
- 🎯 **Stay focused** - Don't get lost in unnecessary details
- 😊 **Stay calm** - Take your time to think through problems

**Good luck crushing those interviews! 🚀💪** evolve independently while maintaining well-defined interfaces.*

### **Section 2: Data Storage and Management (Questions 9-16)**

**9. Why does TinyURL prefer NoSQL databases over relational databases?**
- A) NoSQL provides better data consistency
- B) NoSQL offers better horizontal scalability for simple key-value operations
- C) NoSQL has better SQL query support
- D) NoSQL provides automatic data compression

**Answer: B) NoSQL offers better horizontal scalability for simple key-value operations**
*Explanation: URL shortening primarily involves simple key-value lookups, which NoSQL databases handle more efficiently at scale.*

**10. Which database property is most important for Uber's payment processing system?**
- A) Eventual consistency
- B) High availability over consistency
- C) Strong consistency (ACID properties)
- D) Partition tolerance over availability

**Answer: C) Strong consistency (ACID properties)**
*Explanation: Financial transactions require strong consistency to ensure accurate payment processing and prevent double-spending.*

**11. What is the primary reason WhatsApp uses Erlang for message processing?**
- A) Erlang provides built-in encryption
- B) Erlang excels at handling massive concurrency with lightweight processes
- C) Erlang has the fastest database drivers
- D) Erlang automatically scales across multiple servers

**Answer: B) Erlang excels at handling massive concurrency with lightweight processes**
*Explanation: Erlang's actor model and lightweight processes make it ideal for handling millions of concurrent connections.*

**12. Which caching strategy is most appropriate for frequently accessed shortened URLs?**
- A) Write-through caching
- B) Write-behind caching
- C) Cache-aside (lazy loading)
- D) Distributed caching with TTL

**Answer: D) Distributed caching with TTL**
*Explanation: Popular URLs should be cached across multiple nodes with appropriate TTL to handle high read volumes.*

**13. How does Airbnb handle property search across millions of listings efficiently?**
- A) Linear search through all properties
- B) Use Elasticsearch with geolocation and filtering capabilities
- C) Store all data in memory for faster access
- D) Use only SQL databases with complex joins

**Answer: B) Use Elasticsearch with geolocation and filtering capabilities**
*Explanation: Elasticsearch provides full-text search, geolocation queries, and complex filtering needed for property discovery.*

**14. What is the primary challenge in designing notification service data storage?**
- A) Handling different message formats across multiple channels
- B) Storing user preferences and managing opt-in/opt-out efficiently
- C) Encrypting all notification content
- D) Compressing notification data for storage

**Answer: B) Storing user preferences and managing opt-in/opt-out efficiently**
*Explanation: User preferences for different notification types and channels must be efficiently stored and quickly accessible.*

**15. Which data partitioning strategy is most effective for Zoom's video conferencing system?**
- A) Partition by user ID hash
- B) Partition by meeting ID and geographic region
- C) Partition by timestamp
- D) Use no partitioning, store everything centrally

**Answer: B) Partition by meeting ID and geographic region**
*Explanation: Partitioning by meeting and region keeps related data together and reduces latency for participants.*

**16. What is the main advantage of using event sourcing in Uber's trip management?**
- A) Reduces storage requirements
- B) Provides complete audit trail and enables event replay
- C) Simplifies database schema design
- D) Eliminates the need for backup systems

**Answer: B) Provides complete audit trail and enables event replay**
*Explanation: Event sourcing maintains a complete history of trip events, enabling auditing and system recovery.*

### **Section 3: Real-time Systems and Communication (Questions 17-24)**

**17. Which protocol is most suitable for real-time messaging in WhatsApp?**
- A) HTTP with long polling
- B) WebSockets for persistent connections
- C) REST API with frequent polling
- D) FTP for file transfer

**Answer: B) WebSockets for persistent connections**
*Explanation: WebSockets maintain persistent connections enabling real-time, bidirectional communication with minimal overhead.*

**18. How does Google Maps achieve sub-second route calculation for millions of users?**
- A) Use only real-time computation for all routes
- B) Precompute popular routes and use caching with graph algorithms
- C) Store all possible routes in advance
- D) Use linear search through all road segments

**Answer: B) Precompute popular routes and use caching with graph algorithms**
*Explanation: Combining precomputed routes for popular paths with real-time graph algorithms and caching provides optimal performance.*

**19. What is the primary challenge in Zoom's real-time video processing?**
- A) Video compression algorithms
- B) Managing bandwidth adaptation and latency optimization
- C) User interface design
- D) File storage optimization

**Answer: B) Managing bandwidth adaptation and latency optimization**
*Explanation: Real-time video requires adaptive bitrate streaming and latency optimization for quality user experience.*

**20. Which approach best handles Twitter's real-time trending topics calculation?**
- A) Batch processing every hour
- B) Stream processing with sliding window algorithms
- C) Manual curation by content moderators
- D) Simple counting without time considerations

**Answer: B) Stream processing with sliding window algorithms**
*Explanation: Real-time trending requires continuous processing of tweet streams with time-windowed analysis.*

**21. How does Facebook ensure real-time news feed updates for billions of users?**
- A) Periodic polling every minute
- B) Event-driven architecture with push notifications and fan-out
- C) Manual refresh by users
- D) Batch processing every few hours

**Answer: B) Event-driven architecture with push notifications and fan-out**
*Explanation: Event-driven systems with intelligent fan-out strategies enable real-time updates at massive scale.*

**22. What is the main requirement for Netflix's video streaming to handle millions of concurrent viewers?**
- A) Single high-performance server
- B) Adaptive bitrate streaming with global CDN
- C) Store all videos in a single data center
- D) Use only the highest quality video format

**Answer: B) Adaptive bitrate streaming with global CDN**
*Explanation: Adaptive streaming adjusts quality based on user bandwidth while CDN ensures global content availability.*

**23. Which messaging pattern is most appropriate for notification service delivery?**
- A) Synchronous request-response
- B) Asynchronous message queues with retry logic
- C) Direct database polling
- D) File-based communication

**Answer: B) Asynchronous message queues with retry logic**
*Explanation: Asynchronous queues decouple services and provide reliability through retry mechanisms for failed deliveries.*

**24. How does Uber achieve real-time driver-rider matching within seconds?**
- A) Check all drivers sequentially
- B) Use geospatial indexing with proximity algorithms
- C) Random assignment within city limits
- D) Manual dispatcher assignment

**Answer: B) Use geospatial indexing with proximity algorithms**
*Explanation: Spatial data structures enable quick location queries to find nearby drivers efficiently.*

### **Section 4: System Design Principles and Trade-offs (Questions 25-32)**

**25. According to the CAP theorem, what trade-off must distributed systems make?**
- A) Choose between speed and accuracy
- B) Balance consistency, availability, and partition tolerance (can't have all three)
- C) Optimize for either read or write performance
- D) Select between security and performance

**Answer: B) Balance consistency, availability, and partition tolerance (can't have all three)**
*Explanation: CAP theorem states that distributed systems can only guarantee two of the three properties simultaneously.*

**26. What is the primary trade-off when implementing Amazon's eventual consistency model?**
- A) Higher cost vs. lower performance
- B) Immediate consistency vs. higher availability and partition tolerance
- C) Security vs. usability
- D) Complexity vs. simplicity

**Answer: B) Immediate consistency vs. higher availability and partition tolerance**
*Explanation: Eventual consistency sacrifices immediate consistency to achieve better availability and fault tolerance.*

**27. Which principle guides WhatsApp's architecture of handling 100 billion messages daily with minimal infrastructure?**
- A) Over-engineering for future requirements
- B) Complexity through advanced algorithms
- C) Simplicity and efficiency in design
- D) Using the latest technology trends

**Answer: C) Simplicity and efficiency in design**
*Explanation: WhatsApp's success comes from focusing on core functionality with simple, efficient architecture.*

**28. What is the main reason for implementing circuit breakers in Netflix's microservices?**
- A) Improve system performance
- B) Prevent cascading failures across services
- C) Reduce network bandwidth usage
- D) Simplify service deployment

**Answer: B) Prevent cascading failures across services**
*Explanation: Circuit breakers stop requests to failing services, preventing failures from propagating through the system.*

**29. Why does TinyURL use Base62 encoding for short URLs instead of Base64?**
- A) Base62 provides better compression
- B) Base62 is faster to compute
- C) Base62 uses URL-safe characters (avoids +, /, =)
- D) Base62 provides better security

**Answer: C) Base62 uses URL-safe characters (avoids +, /, =)**
*Explanation: Base62 uses only alphanumeric characters, making URLs safe for web usage without encoding issues.*

**30. What is the primary consideration when designing Airbnb's global payment processing system?**
- A) Processing speed over accuracy
- B) Regulatory compliance and multi-currency support
- C) Minimizing transaction fees
- D) Supporting only credit card payments

**Answer: B) Regulatory compliance and multi-currency support**
*Explanation: Global payment systems must comply with various financial regulations and handle multiple currencies.*

**31. Which approach best balances performance and cost in Google Maps' tile serving system?**
- A) Generate all tiles in real-time
- B) Pre-generate and cache tiles at multiple zoom levels with CDN
- C) Store only the highest resolution tiles
- D) Use vector graphics for all map rendering

**Answer: B) Pre-generate and cache tiles at multiple zoom levels with CDN**
*Explanation: Pre-generated tiles with CDN caching provide fast access while managing storage costs effectively.*

**32. What is the key architectural decision that enables Zoom to support 1000+ participants in a single meeting?**
- A) Use peer-to-peer connections between all participants
- B) Implement selective forwarding units (SFUs) with media routing
- C) Require all participants to have high-bandwidth connections
- D) Limit video quality to reduce bandwidth requirements

**Answer: B) Implement selective forwarding units (SFUs) with media routing**
*Explanation: SFUs efficiently route media streams, reducing bandwidth requirements compared to mesh topology.*

---

## Part B: Real-World Interview Questions (10 Questions)

### **Question 1: Design a URL Shortening Service (TinyURL)**
**Scenario**: You need to design a URL shortening service that can handle 100 million URLs being shortened per day, with a read-to-write ratio of 100:1.

**Key Discussion Points**:
- How would you generate unique short URLs? Compare different approaches (counter-based, hash-based, random generation)
- How would you handle the database design and what type of database would you choose?
- What caching strategy would you implement for high read throughput?
- How would you handle analytics and tracking click metrics?
- What are the scalability bottlenecks and how would you address them?

**Follow-up**: How would you implement custom aliases and URL expiration?

---

### **Question 2: Design Netflix's Video Streaming Architecture**
**Scenario**: Design a global video streaming platform that can serve 100 million users worldwide with high-quality video content.

**Key Discussion Points**:
- How would you handle video storage, encoding, and delivery?
- What is your CDN strategy for global content distribution?
- How would you implement adaptive bitrate streaming?
- How would you design the recommendation system?
- What are your strategies for handling peak traffic (e.g., popular show releases)?

**Follow-up**: How would you implement offline viewing and content pre-loading?

---

### **Question 3: Design Facebook's News Feed System**
**Scenario**: Design a news feed system that can handle 3 billion users posting and consuming content in real-time.

**Key Discussion Points**:
- How would you approach the fan-out problem (push vs. pull vs. hybrid)?
- How would you handle the celebrity user problem?
- What is your strategy for content ranking and personalization?
- How would you design the data model for posts, relationships, and feeds?
- How would you ensure real-time updates?

**Follow-up**: How would you implement content moderation at scale?

---

### **Question 4: Design Uber's Ride Matching System**
**Scenario**: Design a system that can match millions of riders with drivers in real-time across multiple cities globally.

**Key Discussion Points**:
- How would you handle real-time location tracking and updates?
- What data structures would you use for efficient proximity searches?
- How would you implement the matching algorithm considering factors like distance, price, driver ratings?
- How would you handle high-demand scenarios (surge pricing)?
- What is your approach to handling trip state management?

**Follow-up**: How would you implement ride sharing (UberPool) with multiple passengers?

---

### **Question 5: Design WhatsApp's Messaging System**
**Scenario**: Design a messaging system that can handle 100 billion messages per day with end-to-end encryption.

**Key Discussion Points**:
- How would you ensure message delivery reliability?
- What is your approach to handling online/offline users?
- How would you implement group messaging efficiently?
- How would you design the protocol for mobile optimization?
- What is your strategy for handling media messages (images, videos)?

**Follow-up**: How would you implement message synchronization across multiple devices?

---

### **Question 6: Design Google Maps' Navigation System**
**Scenario**: Design a navigation system that provides real-time routing and traffic information to millions of users globally.

**Key Discussion Points**:
- How would you model and store road network data?
- What algorithms would you use for route calculation?
- How would you collect and process real-time traffic data?
- How would you handle route recalculation during navigation?
- What is your caching strategy for maps and routes?

**Follow-up**: How would you implement ETA prediction with machine learning?

---

### **Question 7: Design Amazon's Product Recommendation System**
**Scenario**: Design a recommendation system that can suggest relevant products to 300 million users based on their browsing and purchase history.

**Key Discussion Points**:
- What recommendation algorithms would you implement (collaborative filtering, content-based, hybrid)?
- How would you handle the cold start problem for new users/products?
- How would you design the data pipeline for real-time recommendations?
- What is your approach to handling scale and serving recommendations quickly?
- How would you measure and improve recommendation quality?

**Follow-up**: How would you handle seasonal trends and trending products?

---

### **Question 8: Design Airbnb's Search and Booking System**
**Scenario**: Design a system that allows users to search for accommodations and make bookings, handling millions of properties and bookings globally.

**Key Discussion Points**:
- How would you design the search functionality with multiple filters (location, price, amenities)?
- How would you handle availability checking and booking conflicts?
- What is your approach to handling different time zones and currencies?
- How would you implement the review and rating system?
- How would you design the pricing engine for dynamic pricing?

**Follow-up**: How would you handle fraud detection and trust & safety?

---

### **Question 9: Design a Global Notification Service**
**Scenario**: Design a notification service that can send notifications through multiple channels (email, SMS, push, in-app) to millions of users across different applications.

**Key Discussion Points**:
- How would you design the system to support multiple notification channels?
- How would you handle user preferences and opt-in/opt-out management?
- What is your approach to ensuring delivery reliability and handling failures?
- How would you implement rate limiting to respect external service limits?
- How would you design the system for multi-tenancy (serving multiple client applications)?

**Follow-up**: How would you implement notification scheduling and batching?

---

### **Question 10: Design Zoom's Video Conferencing System**
**Scenario**: Design a video conferencing system that can support meetings with up to 1000 participants with features like screen sharing and recording.

**Key Discussion Points**:
- How would you handle video/audio streaming for large meetings?
- What is your approach to bandwidth optimization and quality adaptation?
- How would you implement screen sharing efficiently?
- How would you design the signaling system for meeting management?
- What is your strategy for handling different device types and network conditions?

**Follow-up**: How would you implement breakout rooms and meeting recording?

---

## Answer Framework for Interview Questions

### **1. Requirements Gathering (5 minutes)**
- **Functional Requirements**: What features does the system need?
- **Non-functional Requirements**: Scale, performance, availability requirements
- **Constraints**: Budget, timeline, technology restrictions

### **2. Capacity Estimation (5 minutes)**
- **Users**: Number of active users (daily/monthly)
- **Data**: Storage requirements and growth projections
- **Bandwidth**: Network requirements for different operations
- **QPS**: Queries per second calculations

### **3. System Design (15-20 minutes)**
- **High-level Architecture**: Major components and their interactions
- **Database Design**: Schema, partitioning, consistency requirements
- **API Design**: Key endpoints and their functionality
- **Deep Dive**: Focus on 1-2 critical components

### **4. Scale and Optimization (10 minutes)**
- **Bottlenecks**: Identify potential failure points
- **Scaling Strategy**: Horizontal vs. vertical scaling approaches
- **Caching**: What and where to cache
- **Monitoring**: Key metrics and alerting

### **5. Trade-offs and Alternatives (5 minutes)**
- **Technology Choices**: Justify your technology selections
- **Consistency vs. Availability**: CAP theorem considerations
- **Performance vs. Cost**: Optimization trade-offs
- **Security**: Authentication, authorization, data protection

---

## Tips for Success

### **Before the Interview**
1. **Study System Components**: Understand load balancers, databases, caching, message queues
2. **Practice Calculations**: Get comfortable with back-of-the-envelope estimations
3. **Know Trade-offs**: Understand when to use different technologies and architectures
4. **Review Real Systems**: Study how major companies solve similar problems

### **During the Interview**
1. **Ask Clarifying Questions**: Understand requirements before designing
2. **Start High-level**: Begin with overall architecture before diving into details
3. **Think Out Loud**: Explain your reasoning and decision-making process
4. **Consider Trade-offs**: Discuss pros and cons of different approaches
5. **Be Realistic**: Acknowledge limitations and areas for improvement

### **Common Mistakes to Avoid**
1. **Jumping to Solutions**: Design without understanding requirements
2. **Over-engineering**: Adding unnecessary complexity
3. **Ignoring Scale**: Not considering performance at the required scale
4. **Single Points of Failure**: Designing systems without redundancy
5. **Neglecting Monitoring**: Not considering observability and maintenance

Remember: System design interviews are conversations, not exams. The interviewer wants to see how you think through complex problems, make trade-offs, and communicate technical concepts clearly.