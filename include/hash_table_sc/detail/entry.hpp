#pragma once

template <class Key, class T>
struct Entry
{
    Key key;
    T value;
    Entry<Key, T> *next;
};