import auxiliary.collection_fields as collection_fields

def is_a_doc_id_identifier(line):
    return (len(line) >= 2) and ((line[0]+line[1]) == collection_fields.DOC_ID)

def is_a_field_identifier(line):
    return (line in collection_fields.ALL) or (is_a_doc_id_identifier(line))

def parse_field(collection):
    field_text = ''
    collection_line = collection.readline()
    
    # get next line until its a field identifier
    while(not is_a_field_identifier(collection_line.rstrip('\n'))):
        field_text += collection_line.replace('\n', ' ')
        collection_line = collection.readline()

    return field_text.strip(' ').replace('  ', ' '), collection_line