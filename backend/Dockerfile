# Build with backend/ as the context - unlike the inference service, this
# one only needs its own pom.xml and src/:
#
#   docker build -f backend/Dockerfile -t basketball-backend backend
#
# Nothing in the container can reach the host as "localhost", so the two
# outbound dependencies are overridden at run time:
#
#   docker run -p 8080:8080 `
#     -e DB_PASSWORD="postgres" `
#     -e DB_HOST="host.docker.internal" `
#     -e INFERENCE_URL="http://host.docker.internal:8000" `
#     basketball-backend
#
# host.docker.internal is Docker Desktop's DNS name for the host machine.
# It is a stopgap: docker-compose will replace it with real service names
# on a shared network.

FROM maven:3.9-eclipse-temurin-21 AS build

WORKDIR /build

# The pom alone first, so the dependency download lands in its own layer
# and editing Java source does not re-resolve the whole tree.
COPY pom.xml .
RUN mvn -B dependency:go-offline

COPY src ./src

# Tests are skipped deliberately, not to save time. contextLoads starts the
# full Spring context, which needs a live Postgres - that exists at run
# time, never during an image build.
RUN mvn -B clean package -DskipTests

FROM eclipse-temurin:21-jre-alpine

WORKDIR /app

# Only the jar crosses the stage boundary; Maven and the JDK stay behind.
COPY --from=build /build/target/*.jar app.jar

EXPOSE 8080

CMD ["java", "-jar", "app.jar"]
