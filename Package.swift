// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "LLM",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
    ],
    products: [
        .library(
            name: "LLM",
            targets: ["LLM"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/swiftlang/swift-docc-plugin", from: "1.4.3"),
        .package(url: "https://github.com/swift-server/async-http-client.git", from: "1.21.0"),
    ],
    targets: [
        .target(
            name: "LLM",
            dependencies: [
                .product(name: "AsyncHTTPClient", package: "async-http-client",
                         condition: .when(platforms: [.linux])),
            ]
        ),
        .testTarget(
            name: "LLMTests",
            dependencies: ["LLM"],
            resources: [
                .copy("Fixtures"),
            ]
        ),
    ]
)
