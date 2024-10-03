%{
#include "y.tab.h"
#include <stdio.h>
#include <stdlib.h>

extern int yylval;
%}

%%

[A-Za-z]              { yylval = yytext[0]; return VAR; }
[0-9]+                { yylval = atoi(yytext); return NUM; }
"-"[0-9]+             { yylval = -atoi(yytext + 1); return NUM; }  /* Capture negative numbers */
[ \t]+                ; /* Ignore whitespace */
\n                    { return '\n'; }
"="                   { return ASSIGN; }
"+"                   { return ADD; }
"-"                   { return SUB; }
"*"                   { return MUL; }
"/"                   { return DIV; }
.                     { printf("Invalid character: %s\n", yytext); }
%%

int yywrap(void) { return 1; }
