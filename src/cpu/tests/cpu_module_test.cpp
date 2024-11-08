#define CONFIG_CATCH_MAIN
#include <catch2/catch_all.hpp>
TEST_CASE( "Basic test", "[basic]" ) {
    REQUIRE( 1 == 1 );
    REQUIRE( 6 == 2 );
}
