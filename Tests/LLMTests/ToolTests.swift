//
//  ToolTests.swift
//  LLMTests
//
//  Tests for JSONSchema, ToolChoice, ToolDefinition, FunctionDefinition, ToolCall, FunctionCall
//

import Foundation
@testable import LLM
import Testing

// MARK: - JSONSchema Tests

@Test func jsonSchema_convenienceInitializers() {
    let stringSchema = LLM.OpenAICompatibleAPI.JSONSchema.string(description: "A name")
    #expect(stringSchema.type == .string)
    #expect(stringSchema.description == "A name")

    let numberSchema = LLM.OpenAICompatibleAPI.JSONSchema.number(description: "A value")
    #expect(numberSchema.type == .number)
    #expect(numberSchema.description == "A value")

    let integerSchema = LLM.OpenAICompatibleAPI.JSONSchema.integer(description: "Count")
    #expect(integerSchema.type == .integer)
    #expect(integerSchema.description == "Count")

    let booleanSchema = LLM.OpenAICompatibleAPI.JSONSchema.boolean(description: "Flag")
    #expect(booleanSchema.type == .boolean)
    #expect(booleanSchema.description == "Flag")
}

@Test func jsonSchema_stringWithEnum() {
    let schema = LLM.OpenAICompatibleAPI.JSONSchema.string(
        description: "Weather unit",
        enum: ["celsius", "fahrenheit"]
    )
    #expect(schema.type == .string)
    #expect(schema.enum == ["celsius", "fahrenheit"])
}

@Test func jsonSchema_array() {
    let itemSchema = LLM.OpenAICompatibleAPI.JSONSchema.string(description: "A tag")
    let arraySchema = LLM.OpenAICompatibleAPI.JSONSchema.array(items: itemSchema, description: "List of tags")

    #expect(arraySchema.type == .array)
    #expect(arraySchema.items == itemSchema)
    #expect(arraySchema.description == "List of tags")
}

@Test func jsonSchema_object() {
    let schema = LLM.OpenAICompatibleAPI.JSONSchema.object(
        properties: [
            "name": .string(description: "The name"),
            "age": .integer(description: "The age"),
        ],
        required: ["name"],
        description: "A person"
    )

    #expect(schema.type == .object)
    #expect(schema.properties?.count == 2)
    #expect(schema.required == ["name"])
    #expect(schema.description == "A person")
}

@Test func jsonSchema_nestedObject() {
    let addressSchema = LLM.OpenAICompatibleAPI.JSONSchema.object(
        properties: [
            "street": .string(),
            "city": .string(),
        ],
        required: ["street", "city"]
    )

    let personSchema = LLM.OpenAICompatibleAPI.JSONSchema.object(
        properties: [
            "name": .string(),
            "address": addressSchema,
        ],
        required: ["name"]
    )

    #expect(personSchema.type == .object)
    #expect(personSchema.properties?["address"]?.type == .object)
    #expect(personSchema.properties?["address"]?.properties?["street"]?.type == .string)
}

@Test func jsonSchema_equality_identicalSchemasEqual() {
    let schema1 = LLM.OpenAICompatibleAPI.JSONSchema.string(description: "test")
    let schema2 = LLM.OpenAICompatibleAPI.JSONSchema.string(description: "test")
    #expect(schema1 == schema2)
}

@Test func jsonSchema_equality_differentSchemaNotEqual() {
    let schema1 = LLM.OpenAICompatibleAPI.JSONSchema.string(description: "test")
    let schema2 = LLM.OpenAICompatibleAPI.JSONSchema.string(description: "different")
    #expect(schema1 != schema2)

    let schema3 = LLM.OpenAICompatibleAPI.JSONSchema.string()
    let schema4 = LLM.OpenAICompatibleAPI.JSONSchema.integer()
    #expect(schema3 != schema4)
}

@Test func jsonSchema_hashable_equalObjectsHaveEqualHashes() {
    let schema1 = LLM.OpenAICompatibleAPI.JSONSchema.object(
        properties: ["name": .string()],
        required: ["name"]
    )
    let schema2 = LLM.OpenAICompatibleAPI.JSONSchema.object(
        properties: ["name": .string()],
        required: ["name"]
    )

    #expect(schema1 == schema2)
    #expect(schema1.hashValue == schema2.hashValue)
}

@Test func jsonSchema_hashable_canBeUsedInSet() {
    let schema1 = LLM.OpenAICompatibleAPI.JSONSchema.string()
    let schema2 = LLM.OpenAICompatibleAPI.JSONSchema.string()
    let schema3 = LLM.OpenAICompatibleAPI.JSONSchema.integer()

    var set = Set<LLM.OpenAICompatibleAPI.JSONSchema>()
    set.insert(schema1)
    set.insert(schema2) // Should not increase count (equal to schema1)
    set.insert(schema3)

    #expect(set.count == 2)
}

@Test func jsonSchema_codable_roundTrip() throws {
    let schema = LLM.OpenAICompatibleAPI.JSONSchema.object(
        properties: [
            "location": .string(description: "City name"),
            "unit": .string(description: "Temperature unit", enum: ["celsius", "fahrenheit"]),
        ],
        required: ["location"]
    )

    let encoder = JSONEncoder()
    let decoder = JSONDecoder()

    let data = try encoder.encode(schema)
    let decoded = try decoder.decode(LLM.OpenAICompatibleAPI.JSONSchema.self, from: data)

    #expect(decoded == schema)
}

@Test func jsonSchema_codable_nestedRoundTrip() throws {
    let schema = LLM.OpenAICompatibleAPI.JSONSchema.object(
        properties: [
            "items": .array(items: .object(
                properties: [
                    "id": .integer(),
                    "name": .string(),
                ]
            )),
        ]
    )

    let encoder = JSONEncoder()
    let decoder = JSONDecoder()

    let data = try encoder.encode(schema)
    let decoded = try decoder.decode(LLM.OpenAICompatibleAPI.JSONSchema.self, from: data)

    #expect(decoded == schema)
}

// MARK: - ToolChoice Tests

@Test func toolChoice_encodeAuto() throws {
    let choice = LLM.OpenAICompatibleAPI.ToolChoice.auto
    let encoder = JSONEncoder()
    let data = try encoder.encode(choice)
    let jsonString = String(data: data, encoding: .utf8)
    #expect(jsonString == "\"auto\"")
}

@Test func toolChoice_encodeRequired() throws {
    let choice = LLM.OpenAICompatibleAPI.ToolChoice.required
    let encoder = JSONEncoder()
    let data = try encoder.encode(choice)
    let jsonString = String(data: data, encoding: .utf8)
    #expect(jsonString == "\"required\"")
}

@Test func toolChoice_encodeNone() throws {
    let choice = LLM.OpenAICompatibleAPI.ToolChoice.none
    let encoder = JSONEncoder()
    let data = try encoder.encode(choice)
    let jsonString = String(data: data, encoding: .utf8)
    #expect(jsonString == "\"none\"")
}

@Test func toolChoice_encodeTool() throws {
    let choice = LLM.OpenAICompatibleAPI.ToolChoice.tool(name: "get_weather")
    let encoder = JSONEncoder()
    encoder.outputFormatting = .sortedKeys
    let data = try encoder.encode(choice)
    let jsonString = try #require(String(data: data, encoding: .utf8))

    // Should produce: {"function":{"name":"get_weather"},"type":"function"}
    #expect(jsonString.contains("\"type\":\"function\""))
    #expect(jsonString.contains("\"function\""))
    #expect(jsonString.contains("\"name\":\"get_weather\""))
}

@Test func toolChoice_decodeFromString() throws {
    let decoder = JSONDecoder()

    let autoData = "\"auto\"".data(using: .utf8)!
    let auto = try decoder.decode(LLM.OpenAICompatibleAPI.ToolChoice.self, from: autoData)
    #expect(auto == .auto)

    let requiredData = "\"required\"".data(using: .utf8)!
    let required = try decoder.decode(LLM.OpenAICompatibleAPI.ToolChoice.self, from: requiredData)
    #expect(required == .required)

    let noneData = "\"none\"".data(using: .utf8)!
    let none = try decoder.decode(LLM.OpenAICompatibleAPI.ToolChoice.self, from: noneData)
    #expect(none == .none)
}

@Test func toolChoice_decodeFromObject() throws {
    let json = """
    {"type":"function","function":{"name":"get_weather"}}
    """
    let decoder = JSONDecoder()
    let choice = try decoder.decode(LLM.OpenAICompatibleAPI.ToolChoice.self, from: #require(json.data(using: .utf8)))

    if case let .tool(name) = choice {
        #expect(name == "get_weather")
    } else {
        Issue.record("Expected .tool case")
    }
}

@Test func toolChoice_roundTrip() throws {
    let choices: [LLM.OpenAICompatibleAPI.ToolChoice] = [
        .auto,
        .required,
        .none,
        .tool(name: "search"),
    ]

    let encoder = JSONEncoder()
    let decoder = JSONDecoder()

    for choice in choices {
        let data = try encoder.encode(choice)
        let decoded = try decoder.decode(LLM.OpenAICompatibleAPI.ToolChoice.self, from: data)
        #expect(decoded == choice)
    }
}

// MARK: - ToolDefinition Tests

@Test func toolDefinition_construction() {
    let params = LLM.OpenAICompatibleAPI.JSONSchema.object(
        properties: ["query": .string()],
        required: ["query"]
    )
    let funcDef = LLM.OpenAICompatibleAPI.FunctionDefinition(
        name: "search",
        description: "Search for information",
        parameters: params
    )
    let toolDef = LLM.OpenAICompatibleAPI.ToolDefinition(function: funcDef)

    #expect(toolDef.type == .function)
    #expect(toolDef.function.name == "search")
    #expect(toolDef.function.description == "Search for information")
}

@Test func toolDefinition_codable_roundTrip() throws {
    let params = LLM.OpenAICompatibleAPI.JSONSchema.object(
        properties: [
            "location": .string(description: "City name"),
            "unit": .string(enum: ["celsius", "fahrenheit"]),
        ],
        required: ["location"]
    )
    let funcDef = LLM.OpenAICompatibleAPI.FunctionDefinition(
        name: "get_weather",
        description: "Get the current weather",
        parameters: params
    )
    let toolDef = LLM.OpenAICompatibleAPI.ToolDefinition(function: funcDef)

    let encoder = JSONEncoder()
    let decoder = JSONDecoder()

    let data = try encoder.encode(toolDef)
    let decoded = try decoder.decode(LLM.OpenAICompatibleAPI.ToolDefinition.self, from: data)

    #expect(decoded.type == toolDef.type)
    #expect(decoded.function.name == toolDef.function.name)
    #expect(decoded.function.description == toolDef.function.description)
    #expect(decoded.function.parameters == toolDef.function.parameters)
}

// MARK: - FunctionDefinition Tests

@Test func functionDefinition_construction() {
    let params = LLM.OpenAICompatibleAPI.JSONSchema.object(properties: [:])
    let funcDef = LLM.OpenAICompatibleAPI.FunctionDefinition(
        name: "test_func",
        description: "A test function",
        parameters: params
    )

    #expect(funcDef.name == "test_func")
    #expect(funcDef.description == "A test function")
    #expect(funcDef.parameters.type == .object)
}

// MARK: - ToolCall Tests

@Test func toolCall_construction() {
    let funcCall = LLM.OpenAICompatibleAPI.FunctionCall(
        name: "get_weather",
        arguments: "{\"location\":\"NYC\"}"
    )
    let toolCall = LLM.OpenAICompatibleAPI.ToolCall(
        id: "call_123",
        function: funcCall
    )

    #expect(toolCall.id == "call_123")
    #expect(toolCall.type == "function")
    #expect(toolCall.function.name == "get_weather")
    #expect(toolCall.function.arguments == "{\"location\":\"NYC\"}")
}

@Test func toolCall_codable_roundTrip() throws {
    let funcCall = LLM.OpenAICompatibleAPI.FunctionCall(
        name: "search",
        arguments: "{\"query\":\"swift testing\"}"
    )
    let toolCall = LLM.OpenAICompatibleAPI.ToolCall(
        id: "call_abc",
        type: "function",
        function: funcCall
    )

    let encoder = JSONEncoder()
    let decoder = JSONDecoder()

    let data = try encoder.encode(toolCall)
    let decoded = try decoder.decode(LLM.OpenAICompatibleAPI.ToolCall.self, from: data)

    #expect(decoded.id == toolCall.id)
    #expect(decoded.type == toolCall.type)
    #expect(decoded.function.name == toolCall.function.name)
    #expect(decoded.function.arguments == toolCall.function.arguments)
}

// MARK: - FunctionCall Tests

@Test func functionCall_construction() {
    let funcCall = LLM.OpenAICompatibleAPI.FunctionCall(
        name: "calculate",
        arguments: "{\"a\":1,\"b\":2}"
    )

    #expect(funcCall.name == "calculate")
    #expect(funcCall.arguments == "{\"a\":1,\"b\":2}")
}

@Test func functionCall_codable_roundTrip() throws {
    let funcCall = LLM.OpenAICompatibleAPI.FunctionCall(
        name: "format_date",
        arguments: "{\"date\":\"2024-01-01\",\"format\":\"iso\"}"
    )

    let encoder = JSONEncoder()
    let decoder = JSONDecoder()

    let data = try encoder.encode(funcCall)
    let decoded = try decoder.decode(LLM.OpenAICompatibleAPI.FunctionCall.self, from: data)

    #expect(decoded.name == funcCall.name)
    #expect(decoded.arguments == funcCall.arguments)
}

// MARK: - ToolType Tests

@Test func toolType_rawValue() {
    #expect(LLM.OpenAICompatibleAPI.ToolType.function.rawValue == "function")
}

// MARK: - SchemaType Tests

@Test func schemaType_rawValues() {
    #expect(LLM.OpenAICompatibleAPI.JSONSchema.SchemaType.object.rawValue == "object")
    #expect(LLM.OpenAICompatibleAPI.JSONSchema.SchemaType.array.rawValue == "array")
    #expect(LLM.OpenAICompatibleAPI.JSONSchema.SchemaType.string.rawValue == "string")
    #expect(LLM.OpenAICompatibleAPI.JSONSchema.SchemaType.number.rawValue == "number")
    #expect(LLM.OpenAICompatibleAPI.JSONSchema.SchemaType.integer.rawValue == "integer")
    #expect(LLM.OpenAICompatibleAPI.JSONSchema.SchemaType.boolean.rawValue == "boolean")
    #expect(LLM.OpenAICompatibleAPI.JSONSchema.SchemaType.null.rawValue == "null")
}
