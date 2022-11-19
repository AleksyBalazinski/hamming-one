template <class Key, class T>
struct Entry
{
    Key key;
    T value;
    Entry<Key, T> *next;
};

template <class Key, class T, class Hash>
class TableCpu
{
public: // for now
    Hash hasher;
    size_t count;
    size_t elements;
    Entry<Key, T> **entries;
    Entry<Key, T> *pool;
    Entry<Key, T> *firstFree;

    TableCpu(int nb_entries, int nb_elements)
    {
        count = nb_entries;
        elements = nb_elements;
        entries = new Entry<Key, T> *[nb_entries]();
        pool = new Entry<Key, T>[nb_elements];
        firstFree = pool;
    }

    Entry<Key, T> **getEntries() { return entries; }

    __host__ __device__ Entry<Key, T> *getBucket(Key key)
    {
        size_t hashValue = hasher(key) % count;

        return entries[hashValue];
    }

    ~TableCpu()
    {
        delete[] entries;
        delete[] pool;
    }
};

template <class Key, class Hash>
void addToTable(Key *keys, TableCpu<Key, int, Hash> &table, int seqLength, int numOfSequences)
{
    int totalLen = numOfSequences * seqLength;
    for (int i = 0; i < totalLen; i++)
    {
        Key key = keys[i];
        int value = i / seqLength;

        size_t hashValue = table.hasher(key) % table.count;
        Entry<Key, int> *location = table.firstFree++;
        location->key = key;
        location->value = value;
        location->next = table.entries[hashValue];
        table.entries[hashValue] = location;
    }
}